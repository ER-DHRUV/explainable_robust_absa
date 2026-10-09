from pathlib import Path
import json
import re

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "results/transformer/distilbert_baseline/explainability"
)

INPUT_FILE = (
    BASE_DIR / "explanation_results.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "aspect_context_analysis"
)

INSTANCE_OUTPUT = (
    OUTPUT_DIR / "aspect_context_by_instance.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "aspect_context_summary.csv"
)

ERROR_TYPE_OUTPUT = (
    OUTPUT_DIR / "aspect_context_by_error_type.csv"
)

CONTEXT_ERRORS_OUTPUT = (
    OUTPUT_DIR / "context_dominant_errors.csv"
)

JSON_OUTPUT = (
    OUTPUT_DIR / "aspect_context_analysis_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def find_aspect_span(sentence, aspect):
    """
    Find the character span of the aspect in the sentence.

    Returns:
        (start, end, method)

    method:
        exact
        normalized
        token_overlap
        not_found
    """

    sentence_original = str(sentence)
    aspect_original = str(aspect)

    sentence_lower = sentence_original.lower()
    aspect_lower = aspect_original.lower().strip()

    # --------------------------------------------------------
    # 1. Exact case-insensitive substring match
    # --------------------------------------------------------

    start = sentence_lower.find(
        aspect_lower
    )

    if start >= 0:
        return (
            start,
            start + len(aspect_lower),
            "exact",
        )

    # --------------------------------------------------------
    # 2. Whitespace-normalized match
    # --------------------------------------------------------

    sentence_norm = re.sub(
        r"\s+",
        " ",
        sentence_lower,
    ).strip()

    aspect_norm = re.sub(
        r"\s+",
        " ",
        aspect_lower,
    ).strip()

    start = sentence_norm.find(
        aspect_norm
    )

    if start >= 0:
        return (
            start,
            start + len(aspect_norm),
            "normalized",
        )

    # --------------------------------------------------------
    # 3. Token-overlap fallback
    # --------------------------------------------------------

    aspect_words = re.findall(
        r"\b\w+\b",
        aspect_lower,
    )

    if not aspect_words:
        return (
            None,
            None,
            "not_found",
        )

    sentence_matches = list(
        re.finditer(
            r"\b\w+\b",
            sentence_lower,
        )
    )

    sentence_words = [
        match.group(0)
        for match in sentence_matches
    ]

    if not sentence_words:
        return (
            None,
            None,
            "not_found",
        )

    # --------------------------------------------------------
    # Try contiguous sequence
    # --------------------------------------------------------

    for i in range(
        len(sentence_words)
        - len(aspect_words)
        + 1
    ):

        window = sentence_words[
            i:i + len(aspect_words)
        ]

        if window == aspect_words:

            start_char = (
                sentence_matches[i].start()
            )

            end_char = (
                sentence_matches[
                    i + len(aspect_words) - 1
                ].end()
            )

            return (
                start_char,
                end_char,
                "token_overlap",
            )

    # --------------------------------------------------------
    # Fuzzy positional fallback
    # --------------------------------------------------------

    selected_positions = []

    for word in aspect_words:

        candidates = [
            i
            for i, sentence_word
            in enumerate(sentence_words)
            if sentence_word == word
        ]

        if candidates:
            selected_positions.append(
                candidates[0]
            )

    if selected_positions:

        first_position = min(
            selected_positions
        )

        last_position = max(
            selected_positions
        )

        start_char = (
            sentence_matches[
                first_position
            ].start()
        )

        end_char = (
            sentence_matches[
                last_position
            ].end()
        )

        return (
            start_char,
            end_char,
            "token_overlap",
        )

    return (
        None,
        None,
        "not_found",
    )


def reconstruct_token_spans(
    sentence,
    tokens,
):
    """
    Reconstruct approximate character spans for
    the sequence of SHAP tokens.

    The CSV contains tokenizer pieces rather than
    character offsets.

    We therefore reconstruct their positions
    sequentially in the original sentence.

    Returns:
        list of (start, end)
    """

    sentence = str(sentence)

    sentence_lower = sentence.lower()

    spans = []

    cursor = 0

    for token in tokens:

        if pd.isna(token):

            spans.append(
                (None, None)
            )

            continue

        token = str(token)

        clean = token.strip()

        # Handle WordPiece-style ## tokens.
        clean = clean.replace(
            "##",
            "",
        )

        if not clean:

            spans.append(
                (None, None)
            )

            continue

        clean_lower = clean.lower()

        # ----------------------------------------------------
        # Search from current cursor.
        # ----------------------------------------------------

        pos = sentence_lower.find(
            clean_lower,
            cursor,
        )

        # ----------------------------------------------------
        # Fallback: search from beginning.
        # ----------------------------------------------------

        if pos < 0:

            pos = sentence_lower.find(
                clean_lower
            )

        if pos < 0:

            spans.append(
                (None, None)
            )

            continue

        end = (
            pos
            + len(clean_lower)
        )

        spans.append(
            (pos, end)
        )

        cursor = end

    return spans


def classify_region(
    token_start,
    token_end,
    aspect_start,
    aspect_end,
):
    """
    Determine whether a token overlaps the
    target aspect.

    Returns:
        ASPECT
        CONTEXT
    """

    if (
        token_start is None
        or token_end is None
        or aspect_start is None
        or aspect_end is None
    ):
        return "CONTEXT"

    overlap = (
        token_start < aspect_end
        and token_end > aspect_start
    )

    if overlap:
        return "ASPECT"

    return "CONTEXT"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "Loading SHAP explanations..."
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "sentence_id",
        "aspect",
        "sentence",
        "true_polarity",
        "predicted_polarity",
        "correct",
        "token_index",
        "token",
        "shap_predicted_class",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    print(
        f"Rows      : {len(df):,}"
    )

    print(
        "Instances : "
        f"{df[['sentence_id', 'aspect']]}"
        ".drop_duplicates().shape[0]:,"
    )

    instance_count = (
        df[
            [
                "sentence_id",
                "aspect",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Instances : {instance_count:,}"
    )

    return df


# ============================================================
# INSTANCE ANALYSIS
# ============================================================

def analyze_instances(df):

    print()
    print("=" * 70)
    print("ASPECT vs CONTEXT ATTRIBUTION")
    print("=" * 70)

    instance_rows = []

    grouped = df.groupby(
        [
            "sentence_id",
            "aspect",
        ],
        sort=False,
    )

    total_groups = len(grouped)

    for counter, (
        (sentence_id, aspect),
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):

        sentence = group[
            "sentence"
        ].iloc[0]

        true_polarity = group[
            "true_polarity"
        ].iloc[0]

        predicted_polarity = group[
            "predicted_polarity"
        ].iloc[0]

        correct = bool(
            group["correct"].iloc[0]
        )

        # ----------------------------------------------------
        # Find aspect character span
        # ----------------------------------------------------

        (
            aspect_start,
            aspect_end,
            span_method,
        ) = find_aspect_span(
            sentence,
            aspect,
        )

        # ----------------------------------------------------
        # Sort by SHAP token index
        # ----------------------------------------------------

        group = (
            group
            .sort_values("token_index")
            .copy()
        )

        tokens = group[
            "token"
        ].tolist()

        token_spans = (
            reconstruct_token_spans(
                sentence,
                tokens,
            )
        )

        group["token_start"] = [
            span[0]
            for span in token_spans
        ]

        group["token_end"] = [
            span[1]
            for span in token_spans
        ]

        # ----------------------------------------------------
        # Region assignment
        # ----------------------------------------------------

        group["region"] = [
            classify_region(
                start,
                end,
                aspect_start,
                aspect_end,
            )
            for start, end
            in token_spans
        ]

        # ----------------------------------------------------
        # SHAP values
        # ----------------------------------------------------

        shap_values = pd.to_numeric(
            group[
                "shap_predicted_class"
            ],
            errors="coerce",
        ).fillna(0.0)

        group["abs_shap"] = (
            shap_values.abs()
        )

        aspect_rows = group[
            group["region"] == "ASPECT"
        ]

        context_rows = group[
            group["region"] == "CONTEXT"
        ]

        # ----------------------------------------------------
        # Signed attribution
        # ----------------------------------------------------

        aspect_signed = (
            aspect_rows[
                "shap_predicted_class"
            ].sum()
        )

        context_signed = (
            context_rows[
                "shap_predicted_class"
            ].sum()
        )

        total_signed = (
            shap_values.sum()
        )

        # ----------------------------------------------------
        # Absolute attribution
        # ----------------------------------------------------

        aspect_abs = (
            aspect_rows[
                "abs_shap"
            ].sum()
        )

        context_abs = (
            context_rows[
                "abs_shap"
            ].sum()
        )

        total_abs = (
            shap_values.abs().sum()
        )

        if total_abs > 0:

            aspect_ratio = (
                aspect_abs
                / total_abs
            )

            context_ratio = (
                context_abs
                / total_abs
            )

        else:

            aspect_ratio = 0.0
            context_ratio = 0.0

        # ----------------------------------------------------
        # Dominant evidence
        # ----------------------------------------------------

        if aspect_abs > context_abs:

            dominant_source = "aspect"

        elif context_abs > aspect_abs:

            dominant_source = "context"

        else:

            dominant_source = "tie"

        # ----------------------------------------------------
        # Strongest aspect token
        # ----------------------------------------------------

        strongest_aspect_token = ""
        strongest_aspect_shap = 0.0

        if len(aspect_rows) > 0:

            idx = (
                aspect_rows[
                    "abs_shap"
                ].idxmax()
            )

            strongest_aspect_token = str(
                aspect_rows.loc[
                    idx,
                    "token",
                ]
            )

            strongest_aspect_shap = float(
                aspect_rows.loc[
                    idx,
                    "shap_predicted_class",
                ]
            )

        # ----------------------------------------------------
        # Strongest context token
        # ----------------------------------------------------

        strongest_context_token = ""
        strongest_context_shap = 0.0

        if len(context_rows) > 0:

            idx = (
                context_rows[
                    "abs_shap"
                ].idxmax()
            )

            strongest_context_token = str(
                context_rows.loc[
                    idx,
                    "token",
                ]
            )

            strongest_context_shap = float(
                context_rows.loc[
                    idx,
                    "shap_predicted_class",
                ]
            )

        # ----------------------------------------------------
        # Context dominance strength
        # ----------------------------------------------------

        dominance_difference = (
            context_abs
            - aspect_abs
        )

        # ----------------------------------------------------
        # Token counts
        # ----------------------------------------------------

        aspect_token_count = len(
            aspect_rows
        )

        context_token_count = len(
            context_rows
        )

        # ----------------------------------------------------
        # Store instance-level result
        # ----------------------------------------------------

        instance_rows.append(
            {
                "sentence_id": sentence_id,
                "aspect": aspect,
                "sentence": sentence,
                "true_polarity": true_polarity,
                "predicted_polarity": predicted_polarity,
                "correct": correct,

                "aspect_span_start": (
                    aspect_start
                ),

                "aspect_span_end": (
                    aspect_end
                ),

                "aspect_span_method": (
                    span_method
                ),

                "aspect_token_count": (
                    aspect_token_count
                ),

                "context_token_count": (
                    context_token_count
                ),

                "aspect_signed_shap": (
                    aspect_signed
                ),

                "context_signed_shap": (
                    context_signed
                ),

                "total_signed_shap": (
                    total_signed
                ),

                "aspect_abs_shap": (
                    aspect_abs
                ),

                "context_abs_shap": (
                    context_abs
                ),

                "total_abs_shap": (
                    total_abs
                ),

                "aspect_attribution_ratio": (
                    aspect_ratio
                ),

                "context_attribution_ratio": (
                    context_ratio
                ),

                "dominant_evidence": (
                    dominant_source
                ),

                "context_minus_aspect_abs_shap": (
                    dominance_difference
                ),

                "strongest_aspect_token": (
                    strongest_aspect_token
                ),

                "strongest_aspect_shap": (
                    strongest_aspect_shap
                ),

                "strongest_context_token": (
                    strongest_context_token
                ),

                "strongest_context_shap": (
                    strongest_context_shap
                ),
            }
        )

        if counter % 100 == 0:

            print(
                f"Processed "
                f"{counter:,}/"
                f"{total_groups:,} "
                f"instances"
            )

    return pd.DataFrame(
        instance_rows
    )


# ============================================================
# SUMMARY ANALYSIS
# ============================================================

def create_summary(instance_df):

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    rows = []

    for (
        correct_value,
        group,
    ) in instance_df.groupby(
        "correct",
        dropna=False,
    ):

        label = (
            "correct"
            if bool(correct_value)
            else "incorrect"
        )

        total = len(group)

        aspect_dominant = (
            group[
                "dominant_evidence"
            ] == "aspect"
        ).sum()

        context_dominant = (
            group[
                "dominant_evidence"
            ] == "context"
        ).sum()

        tie = (
            group[
                "dominant_evidence"
            ] == "tie"
        ).sum()

        rows.append(
            {
                "correct": label,
                "instances": total,

                "mean_aspect_abs_shap": (
                    group[
                        "aspect_abs_shap"
                    ].mean()
                ),

                "mean_context_abs_shap": (
                    group[
                        "context_abs_shap"
                    ].mean()
                ),

                "median_aspect_abs_shap": (
                    group[
                        "aspect_abs_shap"
                    ].median()
                ),

                "median_context_abs_shap": (
                    group[
                        "context_abs_shap"
                    ].median()
                ),

                "mean_aspect_attribution_ratio": (
                    group[
                        "aspect_attribution_ratio"
                    ].mean()
                ),

                "mean_context_attribution_ratio": (
                    group[
                        "context_attribution_ratio"
                    ].mean()
                ),

                "aspect_dominant_instances": (
                    aspect_dominant
                ),

                "context_dominant_instances": (
                    context_dominant
                ),

                "tie_instances": tie,

                "context_dominant_rate": (
                    context_dominant
                    / total
                    if total
                    else 0
                ),

                "aspect_dominant_rate": (
                    aspect_dominant
                    / total
                    if total
                    else 0
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return summary


# ============================================================
# ERROR TYPE ANALYSIS
# ============================================================

def create_error_type_analysis(
    instance_df
):

    print()
    print("=" * 70)
    print("ERROR TYPE ANALYSIS")
    print("=" * 70)

    errors = instance_df[
        instance_df["correct"] == False
    ].copy()

    rows = []

    grouped = errors.groupby(
        [
            "true_polarity",
            "predicted_polarity",
        ]
    )

    for (
        true_label,
        pred_label,
    ), group in grouped:

        total = len(group)

        context_dominant = (
            group[
                "dominant_evidence"
            ] == "context"
        ).sum()

        aspect_dominant = (
            group[
                "dominant_evidence"
            ] == "aspect"
        ).sum()

        rows.append(
            {
                "true_polarity": true_label,
                "predicted_polarity": pred_label,
                "instances": total,

                "mean_aspect_abs_shap": (
                    group[
                        "aspect_abs_shap"
                    ].mean()
                ),

                "mean_context_abs_shap": (
                    group[
                        "context_abs_shap"
                    ].mean()
                ),

                "mean_aspect_attribution_ratio": (
                    group[
                        "aspect_attribution_ratio"
                    ].mean()
                ),

                "mean_context_attribution_ratio": (
                    group[
                        "context_attribution_ratio"
                    ].mean()
                ),

                "aspect_dominant_instances": (
                    aspect_dominant
                ),

                "context_dominant_instances": (
                    context_dominant
                ),

                "context_dominant_rate": (
                    context_dominant
                    / total
                    if total
                    else 0
                ),

                "mean_context_minus_aspect": (
                    group[
                        "context_minus_aspect_abs_shap"
                    ].mean()
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if len(result):

        result = result.sort_values(
            "context_dominant_rate",
            ascending=False,
        )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# CONTEXT-DOMINANT ERRORS
# ============================================================

def create_context_dominant_errors(
    instance_df
):

    print()
    print("=" * 70)
    print("CONTEXT-DOMINANT ERRORS")
    print("=" * 70)

    errors = instance_df[
        (instance_df["correct"] == False)
        &
        (
            instance_df[
                "dominant_evidence"
            ] == "context"
        )
    ].copy()

    errors = errors.sort_values(
        "context_minus_aspect_abs_shap",
        ascending=False,
    )

    output_columns = [
        "sentence_id",
        "aspect",
        "sentence",
        "true_polarity",
        "predicted_polarity",
        "aspect_abs_shap",
        "context_abs_shap",
        "aspect_attribution_ratio",
        "context_attribution_ratio",
        "strongest_context_token",
        "strongest_context_shap",
        "strongest_aspect_token",
        "strongest_aspect_shap",
        "context_minus_aspect_abs_shap",
    ]

    errors = errors[
        output_columns
    ]

    print(
        errors.head(20).to_string(
            index=False
        )
    )

    return errors


# ============================================================
# JSON SUMMARY
# ============================================================

def create_json_summary(
    instance_df,
    summary_df,
    error_type_df,
):

    total_instances = len(
        instance_df
    )

    correct_instances = int(
        instance_df["correct"].sum()
    )

    incorrect_instances = (
        total_instances
        - correct_instances
    )

    context_dominant_total = int(
        (
            instance_df[
                "dominant_evidence"
            ] == "context"
        ).sum()
    )

    aspect_dominant_total = int(
        (
            instance_df[
                "dominant_evidence"
            ] == "aspect"
        ).sum()
    )

    incorrect = instance_df[
        instance_df["correct"] == False
    ]

    incorrect_context_dominant = int(
        (
            incorrect[
                "dominant_evidence"
            ] == "context"
        ).sum()
    )

    incorrect_aspect_dominant = int(
        (
            incorrect[
                "dominant_evidence"
            ] == "aspect"
        ).sum()
    )

    summary = {
        "total_instances": (
            total_instances
        ),

        "correct_instances": (
            correct_instances
        ),

        "incorrect_instances": (
            incorrect_instances
        ),

        "accuracy": (
            correct_instances
            / total_instances
            if total_instances
            else 0
        ),

        "overall_context_dominant_instances": (
            context_dominant_total
        ),

        "overall_aspect_dominant_instances": (
            aspect_dominant_total
        ),

        "overall_context_dominant_rate": (
            context_dominant_total
            / total_instances
            if total_instances
            else 0
        ),

        "overall_aspect_dominant_rate": (
            aspect_dominant_total
            / total_instances
            if total_instances
            else 0
        ),

        "incorrect_context_dominant_instances": (
            incorrect_context_dominant
        ),

        "incorrect_aspect_dominant_instances": (
            incorrect_aspect_dominant
        ),

        "incorrect_context_dominant_rate": (
            incorrect_context_dominant
            / incorrect_instances
            if incorrect_instances
            else 0
        ),

        "incorrect_aspect_dominant_rate": (
            incorrect_aspect_dominant
            / incorrect_instances
            if incorrect_instances
            else 0
        ),

        "incorrect_mean_aspect_abs_shap": (
            float(
                incorrect[
                    "aspect_abs_shap"
                ].mean()
            )
            if len(incorrect)
            else 0
        ),

        "incorrect_mean_context_abs_shap": (
            float(
                incorrect[
                    "context_abs_shap"
                ].mean()
            )
            if len(incorrect)
            else 0
        ),

        "incorrect_mean_context_attribution_ratio": (
            float(
                incorrect[
                    "context_attribution_ratio"
                ].mean()
            )
            if len(incorrect)
            else 0
        ),

        "incorrect_mean_aspect_attribution_ratio": (
            float(
                incorrect[
                    "aspect_attribution_ratio"
                ].mean()
            )
            if len(incorrect)
            else 0
        ),
    }

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SHAP ANALYSIS — EXPERIMENT 3")
    print("ASPECT vs CONTEXT ATTRIBUTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Analyze each ABSA instance
    # --------------------------------------------------------

    instance_df = analyze_instances(
        df
    )

    # --------------------------------------------------------
    # Save instance-level results
    # --------------------------------------------------------

    instance_df.to_csv(
        INSTANCE_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        f"Saved: {INSTANCE_OUTPUT}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_df = create_summary(
        instance_df
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        f"Saved: {SUMMARY_OUTPUT}"
    )

    # --------------------------------------------------------
    # Error types
    # --------------------------------------------------------

    error_type_df = (
        create_error_type_analysis(
            instance_df
        )
    )

    error_type_df.to_csv(
        ERROR_TYPE_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        f"Saved: {ERROR_TYPE_OUTPUT}"
    )

    # --------------------------------------------------------
    # Context-dominant errors
    # --------------------------------------------------------

    context_errors = (
        create_context_dominant_errors(
            instance_df
        )
    )

    context_errors.to_csv(
        CONTEXT_ERRORS_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        f"Saved: {CONTEXT_ERRORS_OUTPUT}"
    )

    # --------------------------------------------------------
    # JSON summary
    # --------------------------------------------------------

    json_summary = (
        create_json_summary(
            instance_df,
            summary_df,
            error_type_df,
        )
    )

    print()
    print("=" * 70)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 70)

    print(
        json.dumps(
            json_summary,
            indent=4,
        )
    )

    print()
    print("=" * 70)
    print(
        "ASPECT vs CONTEXT ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    print()
    print("Generated files:")

    print(
        " - aspect_context_by_instance.csv"
    )

    print(
        " - aspect_context_summary.csv"
    )

    print(
        " - aspect_context_by_error_type.csv"
    )

    print(
        " - context_dominant_errors.csv"
    )

    print(
        " - aspect_context_analysis_summary.json"
    )


if __name__ == "__main__":
    main()
