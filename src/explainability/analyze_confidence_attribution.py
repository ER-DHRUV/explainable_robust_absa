import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "results/transformer/distilbert_baseline/explainability"
)

INPUT_FILE = BASE_DIR / "explanation_results.csv"

OUTPUT_DIR = BASE_DIR / "confidence_attribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

POLARITIES = ["negative", "neutral", "positive"]

SHAP_COLUMNS = {
    "negative": "shap_negative",
    "neutral": "shap_neutral",
    "positive": "shap_positive",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_polarity(value):
    """
    Normalize polarity labels to lowercase strings.
    """
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in POLARITIES:
        return value

    return value


def softmax(values):
    """
    Numerically stable softmax.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    max_value = np.max(values)

    exp_values = np.exp(values - max_value)
    denominator = np.sum(exp_values)

    if denominator == 0:
        return np.ones_like(values) / len(values)

    return exp_values / denominator


def safe_mean(series):
    """
    Return mean while handling empty/non-numeric input.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def safe_median(series):
    """
    Return median while handling empty/non-numeric input.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.median())


def get_competing_class(row):
    """
    Determine the strongest competing class based on
    class-level SHAP-derived scores.

    The predicted class is excluded.
    """
    predicted = row["predicted_polarity"]

    candidates = {
        polarity: row[f"class_score_{polarity}"]
        for polarity in POLARITIES
        if polarity != predicted
    }

    if not candidates:
        return None

    return max(candidates, key=candidates.get)


def calculate_entropy(probabilities):
    """
    Calculate normalized entropy for a 3-class distribution.

    Range:
        0 = completely concentrated
        1 = maximally distributed
    """
    probabilities = np.asarray(probabilities, dtype=float)

    probabilities = probabilities[
        np.isfinite(probabilities) & (probabilities > 0)
    ]

    if len(probabilities) == 0:
        return np.nan

    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    max_entropy = np.log(3)

    if max_entropy == 0:
        return 0.0

    return float(entropy / max_entropy)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SHAP ANALYSIS — EXPERIMENT 6")
print("CONFIDENCE vs ATTRIBUTION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading SHAP explanations...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows      : {len(df):,}")

required_columns = [
    "sentence_id",
    "aspect",
    "sentence",
    "true_polarity",
    "predicted_polarity",
    "correct",
    "token_index",
    "token",
    "shap_negative",
    "shap_neutral",
    "shap_positive",
    "shap_predicted_class",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# NORMALIZE DATA
# ============================================================

df["true_polarity"] = (
    df["true_polarity"]
    .apply(normalize_polarity)
)

df["predicted_polarity"] = (
    df["predicted_polarity"]
    .apply(normalize_polarity)
)

df["correct"] = df["correct"].astype(bool)

for column in [
    "shap_negative",
    "shap_neutral",
    "shap_positive",
    "shap_predicted_class",
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df["token_clean"] = (
    df["token"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# BUILD INSTANCE-LEVEL ATTRIBUTION PROFILES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING INSTANCE-LEVEL ATTRIBUTION PROFILES")
print("=" * 70)

group_columns = [
    "sentence_id",
    "aspect",
    "sentence",
    "true_polarity",
    "predicted_polarity",
    "correct",
]

instance_rows = []

grouped = df.groupby(
    group_columns,
    dropna=False,
    sort=False,
)

total_instances = len(grouped)

for counter, (group_key, group) in enumerate(
    grouped,
    start=1,
):

    (
        sentence_id,
        aspect,
        sentence,
        true_polarity,
        predicted_polarity,
        correct,
    ) = group_key

    # --------------------------------------------------------
    # Class-level SHAP scores
    # --------------------------------------------------------

    class_scores = {}

    for polarity in POLARITIES:

        column = SHAP_COLUMNS[polarity]

        score = (
            pd.to_numeric(
                group[column],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        class_scores[polarity] = float(score)

    # --------------------------------------------------------
    # Predicted class score
    # --------------------------------------------------------

    predicted_score = class_scores.get(
        predicted_polarity,
        np.nan,
    )

    # --------------------------------------------------------
    # Competing class
    # --------------------------------------------------------

    candidate_scores = {
        polarity: score
        for polarity, score in class_scores.items()
        if polarity != predicted_polarity
    }

    if candidate_scores:

        competing_class = max(
            candidate_scores,
            key=candidate_scores.get,
        )

        competing_score = candidate_scores[
            competing_class
        ]

    else:

        competing_class = None
        competing_score = np.nan

    # --------------------------------------------------------
    # Class margin
    # --------------------------------------------------------

    if (
        pd.notna(predicted_score)
        and pd.notna(competing_score)
    ):

        class_margin = (
            predicted_score
            - competing_score
        )

    else:

        class_margin = np.nan

    absolute_class_margin = (
        abs(class_margin)
        if pd.notna(class_margin)
        else np.nan
    )

    # --------------------------------------------------------
    # SHAP-derived softmax distribution
    # --------------------------------------------------------

    score_vector = np.array(
        [
            class_scores["negative"],
            class_scores["neutral"],
            class_scores["positive"],
        ],
        dtype=float,
    )

    probabilities = softmax(score_vector)

    shap_confidence = np.nan
    predicted_probability = np.nan
    competing_probability = np.nan

    if predicted_polarity in POLARITIES:

        predicted_index = POLARITIES.index(
            predicted_polarity
        )

        shap_confidence = float(
            probabilities[predicted_index]
        )

        competing_probabilities = [
            probabilities[i]
            for i in range(len(POLARITIES))
            if i != predicted_index
        ]

        if competing_probabilities:

            competing_probability = float(
                max(competing_probabilities)
            )

    # --------------------------------------------------------
    # Probability margin
    # --------------------------------------------------------

    if (
        pd.notna(predicted_probability)
        and pd.notna(competing_probability)
    ):

        probability_margin = (
            predicted_probability
            - competing_probability
        )

    else:

        probability_margin = np.nan

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    normalized_entropy = calculate_entropy(
        probabilities
    )

    # --------------------------------------------------------
    # Attribution concentration
    # --------------------------------------------------------

    token_attributions = (
        pd.to_numeric(
            group["shap_predicted_class"],
            errors="coerce",
        )
        .fillna(0)
        .abs()
        .values
    )

    total_abs_attribution = float(
        np.sum(token_attributions)
    )

    if total_abs_attribution > 0:

        sorted_attributions = np.sort(
            token_attributions
        )[::-1]

        top1_share = (
            sorted_attributions[0]
            / total_abs_attribution
            if len(sorted_attributions) > 0
            else np.nan
        )

        top3_share = (
            np.sum(sorted_attributions[:3])
            / total_abs_attribution
            if len(sorted_attributions) > 0
            else np.nan
        )

    else:

        top1_share = np.nan
        top3_share = np.nan

    # --------------------------------------------------------
    # Context vs aspect attribution
    #
    # Aspect attribution is approximated by exact token
    # matching against the aspect string.
    # --------------------------------------------------------

    aspect_text = (
        str(aspect).strip().lower()
        if not pd.isna(aspect)
        else ""
    )

    aspect_tokens = set(
        aspect_text.split()
    )

    token_values = (
        group["token_clean"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    is_aspect_token = token_values.apply(
        lambda token: (
            token in aspect_tokens
            if token
            else False
        )
    )

    aspect_abs_shap = float(
        group.loc[
            is_aspect_token,
            "shap_predicted_class",
        ]
        .abs()
        .sum()
    )

    context_abs_shap = float(
        group.loc[
            ~is_aspect_token,
            "shap_predicted_class",
        ]
        .abs()
        .sum()
    )

    total_aspect_context = (
        aspect_abs_shap
        + context_abs_shap
    )

    if total_aspect_context > 0:

        aspect_attribution_ratio = (
            aspect_abs_shap
            / total_aspect_context
        )

        context_attribution_ratio = (
            context_abs_shap
            / total_aspect_context
        )

    else:

        aspect_attribution_ratio = np.nan
        context_attribution_ratio = np.nan

    attribution_gap = (
        context_attribution_ratio
        - aspect_attribution_ratio
        if (
            pd.notna(context_attribution_ratio)
            and pd.notna(
                aspect_attribution_ratio
            )
        )
        else np.nan
    )

    # --------------------------------------------------------
    # Strongest token
    # --------------------------------------------------------

    strongest_token = None
    strongest_token_shap = np.nan

    if len(group) > 0:

        abs_values = (
            group["shap_predicted_class"]
            .abs()
        )

        if abs_values.notna().any():

            strongest_index = abs_values.idxmax()

            strongest_token = group.loc[
                strongest_index,
                "token_clean",
            ]

            strongest_token_shap = float(
                group.loc[
                    strongest_index,
                    "shap_predicted_class",
                ]
            )

    # --------------------------------------------------------
    # Store instance
    # --------------------------------------------------------

    row = {
        "sentence_id": sentence_id,
        "aspect": aspect,
        "sentence": sentence,
        "true_polarity": true_polarity,
        "predicted_polarity": predicted_polarity,
        "correct": correct,

        "class_score_negative": (
            class_scores["negative"]
        ),
        "class_score_neutral": (
            class_scores["neutral"]
        ),
        "class_score_positive": (
            class_scores["positive"]
        ),

        "predicted_class_score": predicted_score,

        "competing_class": competing_class,
        "competing_class_score": competing_score,

        "class_margin": class_margin,
        "absolute_class_margin": (
            absolute_class_margin
        ),

        "shap_confidence": shap_confidence,
        "competing_probability": (
            competing_probability
        ),
        "probability_margin": (
            probability_margin
        ),

        "normalized_entropy": (
            normalized_entropy
        ),

        "top1_attribution_share": (
            top1_share
        ),
        "top3_attribution_share": (
            top3_share
        ),

        "aspect_abs_shap": aspect_abs_shap,
        "context_abs_shap": context_abs_shap,

        "aspect_attribution_ratio": (
            aspect_attribution_ratio
        ),
        "context_attribution_ratio": (
            context_attribution_ratio
        ),

        "attribution_gap": attribution_gap,

        "strongest_token": strongest_token,
        "strongest_token_shap": (
            strongest_token_shap
        ),

        "token_count": len(group),
    }

    instance_rows.append(row)

    if counter % 100 == 0:
        print(
            f"Processed "
            f"{counter:,}/{total_instances:,} "
            f"instances"
        )


instance_df = pd.DataFrame(instance_rows)

instance_output = (
    OUTPUT_DIR
    / "confidence_attribution_by_instance.csv"
)

instance_df.to_csv(
    instance_output,
    index=False,
)

print(f"\nSaved: {instance_output}")


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OVERALL CONFIDENCE-ATTRIBUTION SUMMARY")
print("=" * 70)

summary = {
    "total_instances": int(
        len(instance_df)
    ),

    "correct_instances": int(
        instance_df["correct"].sum()
    ),

    "incorrect_instances": int(
        (~instance_df["correct"]).sum()
    ),

    "accuracy": float(
        instance_df["correct"].mean()
    ),

    "mean_shap_confidence": safe_mean(
        instance_df["shap_confidence"]
    ),

    "median_shap_confidence": safe_median(
        instance_df["shap_confidence"]
    ),

    "mean_probability_margin": safe_mean(
        instance_df["probability_margin"]
    ),

    "median_probability_margin": safe_median(
        instance_df["probability_margin"]
    ),

    "mean_absolute_class_margin": safe_mean(
        instance_df["absolute_class_margin"]
    ),

    "median_absolute_class_margin": safe_median(
        instance_df["absolute_class_margin"]
    ),

    "mean_normalized_entropy": safe_mean(
        instance_df["normalized_entropy"]
    ),

    "mean_top1_attribution_share": safe_mean(
        instance_df["top1_attribution_share"]
    ),

    "mean_top3_attribution_share": safe_mean(
        instance_df["top3_attribution_share"]
    ),

    "mean_context_attribution_ratio": safe_mean(
        instance_df["context_attribution_ratio"]
    ),

    "mean_aspect_attribution_ratio": safe_mean(
        instance_df["aspect_attribution_ratio"]
    ),

    "mean_attribution_gap": safe_mean(
        instance_df["attribution_gap"]
    ),
}

summary_df = pd.DataFrame([summary])

summary_output = (
    OUTPUT_DIR
    / "confidence_attribution_summary.csv"
)

summary_df.to_csv(
    summary_output,
    index=False,
)

print(
    summary_df.to_string(
        index=False
    )
)

print(f"\nSaved: {summary_output}")


# ============================================================
# CORRECT VS INCORRECT
# ============================================================

print("\n" + "=" * 70)
print("CORRECT VS INCORRECT")
print("=" * 70)

correct_rows = []

for correct_value, group in instance_df.groupby(
    "correct",
    dropna=False,
):

    correct_rows.append(
        {
            "correct": bool(correct_value),
            "instances": int(len(group)),

            "mean_shap_confidence": safe_mean(
                group["shap_confidence"]
            ),

            "median_shap_confidence": safe_median(
                group["shap_confidence"]
            ),

            "mean_probability_margin": safe_mean(
                group["probability_margin"]
            ),

            "median_probability_margin": safe_median(
                group["probability_margin"]
            ),

            "mean_absolute_class_margin": safe_mean(
                group["absolute_class_margin"]
            ),

            "mean_normalized_entropy": safe_mean(
                group["normalized_entropy"]
            ),

            "mean_top1_attribution_share": safe_mean(
                group["top1_attribution_share"]
            ),

            "mean_top3_attribution_share": safe_mean(
                group["top3_attribution_share"]
            ),

            "mean_context_attribution_ratio": (
                safe_mean(
                    group[
                        "context_attribution_ratio"
                    ]
                )
            ),

            "mean_aspect_attribution_ratio": (
                safe_mean(
                    group[
                        "aspect_attribution_ratio"
                    ]
                )
            ),

            "mean_attribution_gap": safe_mean(
                group["attribution_gap"]
            ),
        }
    )


correctness_df = pd.DataFrame(
    correct_rows
)

correctness_output = (
    OUTPUT_DIR
    / "confidence_attribution_by_correctness.csv"
)

correctness_df.to_csv(
    correctness_output,
    index=False,
)

print(
    correctness_df.to_string(
        index=False
    )
)

print(
    f"\nSaved: {correctness_output}"
)


# ============================================================
# CONFIDENCE BINS
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE BIN ANALYSIS")
print("=" * 70)

bins = [
    0.0,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.000001,
]

labels = [
    "<0.40",
    "0.40-0.49",
    "0.50-0.59",
    "0.60-0.69",
    "0.70-0.79",
    "0.80-0.89",
    "0.90-1.00",
]

instance_df["confidence_bin"] = pd.cut(
    instance_df["shap_confidence"],
    bins=bins,
    labels=labels,
    right=False,
    include_lowest=True,
)

confidence_rows = []

for confidence_bin, group in instance_df.groupby(
    "confidence_bin",
    observed=False,
):

    if len(group) == 0:
        continue

    confidence_rows.append(
        {
            "confidence_bin": str(
                confidence_bin
            ),

            "instances": int(len(group)),

            "correct_instances": int(
                group["correct"].sum()
            ),

            "incorrect_instances": int(
                (~group["correct"]).sum()
            ),

            "accuracy": safe_mean(
                group["correct"]
            ),

            "mean_shap_confidence": safe_mean(
                group["shap_confidence"]
            ),

            "mean_probability_margin": safe_mean(
                group["probability_margin"]
            ),

            "mean_absolute_class_margin": safe_mean(
                group["absolute_class_margin"]
            ),

            "mean_normalized_entropy": safe_mean(
                group["normalized_entropy"]
            ),

            "mean_top1_attribution_share": safe_mean(
                group["top1_attribution_share"]
            ),

            "mean_top3_attribution_share": safe_mean(
                group["top3_attribution_share"]
            ),

            "mean_context_attribution_ratio": (
                safe_mean(
                    group[
                        "context_attribution_ratio"
                    ]
                )
            ),

            "mean_aspect_attribution_ratio": (
                safe_mean(
                    group[
                        "aspect_attribution_ratio"
                    ]
                )
            ),

            "mean_attribution_gap": safe_mean(
                group["attribution_gap"]
            ),
        }
    )


confidence_df = pd.DataFrame(
    confidence_rows
)

confidence_output = (
    OUTPUT_DIR
    / "confidence_attribution_by_confidence_bin.csv"
)

confidence_df.to_csv(
    confidence_output,
    index=False,
)

print(
    confidence_df.to_string(
        index=False
    )
)

print(
    f"\nSaved: {confidence_output}"
)


# ============================================================
# ERROR TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ERROR TYPE ANALYSIS")
print("=" * 70)

errors = instance_df[
    instance_df["correct"] == False
].copy()

error_rows = []

if len(errors) > 0:

    error_grouped = errors.groupby(
        [
            "true_polarity",
            "predicted_polarity",
        ],
        dropna=False,
    )

    for (
        true_polarity,
        predicted_polarity,
    ), group in error_grouped:

        error_rows.append(
            {
                "true_polarity": true_polarity,
                "predicted_polarity": (
                    predicted_polarity
                ),

                "instances": int(
                    len(group)
                ),

                "mean_shap_confidence": safe_mean(
                    group["shap_confidence"]
                ),

                "median_shap_confidence": safe_median(
                    group["shap_confidence"]
                ),

                "mean_probability_margin": safe_mean(
                    group["probability_margin"]
                ),

                "mean_absolute_class_margin": (
                    safe_mean(
                        group[
                            "absolute_class_margin"
                        ]
                    )
                ),

                "mean_normalized_entropy": safe_mean(
                    group["normalized_entropy"]
                ),

                "mean_top1_attribution_share": (
                    safe_mean(
                        group[
                            "top1_attribution_share"
                        ]
                    )
                ),

                "mean_top3_attribution_share": (
                    safe_mean(
                        group[
                            "top3_attribution_share"
                        ]
                    )
                ),

                "mean_context_attribution_ratio": (
                    safe_mean(
                        group[
                            "context_attribution_ratio"
                        ]
                    )
                ),

                "mean_aspect_attribution_ratio": (
                    safe_mean(
                        group[
                            "aspect_attribution_ratio"
                        ]
                    )
                ),

                "mean_attribution_gap": safe_mean(
                    group["attribution_gap"]
                ),
            }
        )


error_type_df = pd.DataFrame(
    error_rows
)

error_type_output = (
    OUTPUT_DIR
    / "confidence_attribution_by_error_type.csv"
)

error_type_df.to_csv(
    error_type_output,
    index=False,
)

if len(error_type_df) > 0:
    print(
        error_type_df.to_string(
            index=False
        )
    )

print(
    f"\nSaved: {error_type_output}"
)


# ============================================================
# LOW-CONFIDENCE ERRORS
# ============================================================

print("\n" + "=" * 70)
print("LOW-CONFIDENCE ERRORS")
print("=" * 70)

low_confidence_errors = (
    errors
    .sort_values(
        by="shap_confidence",
        ascending=True,
    )
    .head(50)
    .copy()
)

low_confidence_columns = [
    "sentence_id",
    "aspect",
    "true_polarity",
    "predicted_polarity",
    "shap_confidence",
    "probability_margin",
    "absolute_class_margin",
    "normalized_entropy",
    "top1_attribution_share",
    "top3_attribution_share",
    "context_attribution_ratio",
    "attribution_gap",
    "strongest_token",
    "strongest_token_shap",
]

low_confidence_errors = (
    low_confidence_errors[
        low_confidence_columns
    ]
)

low_confidence_output = (
    OUTPUT_DIR
    / "low_confidence_errors.csv"
)

low_confidence_errors.to_csv(
    low_confidence_output,
    index=False,
)

print(
    low_confidence_errors.to_string(
        index=False
    )
)

print(
    f"\nSaved: {low_confidence_output}"
)


# ============================================================
# HIGH-CONFIDENCE ERRORS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE ERRORS")
print("=" * 70)

high_confidence_errors = (
    errors
    .sort_values(
        by="shap_confidence",
        ascending=False,
    )
    .head(50)
    .copy()
)

high_confidence_errors = (
    high_confidence_errors[
        low_confidence_columns
    ]
)

high_confidence_output = (
    OUTPUT_DIR
    / "high_confidence_errors.csv"
)

high_confidence_errors.to_csv(
    high_confidence_output,
    index=False,
)

print(
    high_confidence_errors.to_string(
        index=False
    )
)

print(
    f"\nSaved: {high_confidence_output}"
)


# ============================================================
# HIGH CONFIDENCE + HIGH ATTRIBUTION CONCENTRATION
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE / HIGH-CONCENTRATION ERRORS")
print("=" * 70)

if len(errors) > 0:

    concentrated_errors = errors[
        (
            errors["shap_confidence"] >= 0.60
        )
        &
        (
            errors["top1_attribution_share"]
            >= 0.30
        )
    ].copy()

    concentrated_errors = (
        concentrated_errors
        .sort_values(
            by=[
                "shap_confidence",
                "top1_attribution_share",
            ],
            ascending=False,
        )
        .head(50)
    )

else:

    concentrated_errors = pd.DataFrame(
        columns=low_confidence_columns
    )


concentration_output = (
    OUTPUT_DIR
    / "high_confidence_high_concentration_errors.csv"
)

concentrated_errors[
    low_confidence_columns
].to_csv(
    concentration_output,
    index=False,
)

if len(concentrated_errors) > 0:

    print(
        concentrated_errors[
            low_confidence_columns
        ].to_string(index=False)
    )

else:

    print(
        "No errors satisfied the "
        "high-confidence/high-concentration "
        "criteria."
    )

print(
    f"\nSaved: {concentration_output}"
)


# ============================================================
# CONFIDENCE-ATTRIBUTION CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE vs ATTRIBUTION CORRELATIONS")
print("=" * 70)

correlation_pairs = {
    "confidence_vs_top1_share": (
        "shap_confidence",
        "top1_attribution_share",
    ),

    "confidence_vs_top3_share": (
        "shap_confidence",
        "top3_attribution_share",
    ),

    "confidence_vs_entropy": (
        "shap_confidence",
        "normalized_entropy",
    ),

    "confidence_vs_context_ratio": (
        "shap_confidence",
        "context_attribution_ratio",
    ),

    "confidence_vs_aspect_ratio": (
        "shap_confidence",
        "aspect_attribution_ratio",
    ),

    "confidence_vs_attribution_gap": (
        "shap_confidence",
        "attribution_gap",
    ),

    "confidence_vs_class_margin": (
        "shap_confidence",
        "absolute_class_margin",
    ),
}

correlation_rows = []

for name, (
    x_column,
    y_column,
) in correlation_pairs.items():

    valid = instance_df[
        [
            x_column,
            y_column,
        ]
    ].dropna()

    if len(valid) >= 2:

        pearson = valid[
            x_column
        ].corr(
            valid[y_column],
            method="pearson",
        )

        spearman = valid[
            x_column
        ].corr(
            valid[y_column],
            method="spearman",
        )

    else:

        pearson = np.nan
        spearman = np.nan

    correlation_rows.append(
        {
            "relationship": name,
            "x_variable": x_column,
            "y_variable": y_column,
            "instances": int(len(valid)),

            "pearson_correlation": (
                float(pearson)
                if pd.notna(pearson)
                else np.nan
            ),

            "spearman_correlation": (
                float(spearman)
                if pd.notna(spearman)
                else np.nan
            ),
        }
    )


correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_output = (
    OUTPUT_DIR
    / "confidence_attribution_correlations.csv"
)

correlation_df.to_csv(
    correlation_output,
    index=False,
)

print(
    correlation_df.to_string(
        index=False
    )
)

print(
    f"\nSaved: {correlation_output}"
)


# ============================================================
# FINAL JSON SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CREATING SUMMARY")
print("=" * 70)

low_confidence_error_count = (
    int(
        (
            errors["shap_confidence"]
            < 0.50
        ).sum()
    )
    if len(errors) > 0
    else 0
)

high_confidence_error_count = (
    int(
        (
            errors["shap_confidence"]
            >= 0.80
        ).sum()
    )
    if len(errors) > 0
    else 0
)

high_concentration_error_count = (
    int(
        (
            errors["top1_attribution_share"]
            >= 0.30
        ).sum()
    )
    if len(errors) > 0
    else 0
)


summary_json = {

    "total_instances": int(
        len(instance_df)
    ),

    "correct_instances": int(
        instance_df["correct"].sum()
    ),

    "incorrect_instances": int(
        (~instance_df["correct"]).sum()
    ),

    "accuracy": float(
        instance_df["correct"].mean()
    ),

    "mean_shap_confidence": safe_mean(
        instance_df["shap_confidence"]
    ),

    "median_shap_confidence": safe_median(
        instance_df["shap_confidence"]
    ),

    "mean_probability_margin": safe_mean(
        instance_df["probability_margin"]
    ),

    "median_probability_margin": safe_median(
        instance_df["probability_margin"]
    ),

    "mean_absolute_class_margin": safe_mean(
        instance_df["absolute_class_margin"]
    ),

    "mean_normalized_entropy": safe_mean(
        instance_df["normalized_entropy"]
    ),

    "mean_top1_attribution_share": safe_mean(
        instance_df["top1_attribution_share"]
    ),

    "mean_top3_attribution_share": safe_mean(
        instance_df["top3_attribution_share"]
    ),

    "mean_context_attribution_ratio": safe_mean(
        instance_df["context_attribution_ratio"]
    ),

    "mean_aspect_attribution_ratio": safe_mean(
        instance_df["aspect_attribution_ratio"]
    ),

    "mean_attribution_gap": safe_mean(
        instance_df["attribution_gap"]
    ),

    "correct_mean_shap_confidence": safe_mean(
        instance_df.loc[
            instance_df["correct"],
            "shap_confidence",
        ]
    ),

    "incorrect_mean_shap_confidence": safe_mean(
        instance_df.loc[
            ~instance_df["correct"],
            "shap_confidence",
        ]
    ),

    "correct_mean_probability_margin": safe_mean(
        instance_df.loc[
            instance_df["correct"],
            "probability_margin",
        ]
    ),

    "incorrect_mean_probability_margin": safe_mean(
        instance_df.loc[
            ~instance_df["correct"],
            "probability_margin",
        ]
    ),

    "correct_mean_absolute_class_margin": safe_mean(
        instance_df.loc[
            instance_df["correct"],
            "absolute_class_margin",
        ]
    ),

    "incorrect_mean_absolute_class_margin": safe_mean(
        instance_df.loc[
            ~instance_df["correct"],
            "absolute_class_margin",
        ]
    ),

    "correct_mean_normalized_entropy": safe_mean(
        instance_df.loc[
            instance_df["correct"],
            "normalized_entropy",
        ]
    ),

    "incorrect_mean_normalized_entropy": safe_mean(
        instance_df.loc[
            ~instance_df["correct"],
            "normalized_entropy",
        ]
    ),

    "correct_mean_top1_attribution_share": safe_mean(
        instance_df.loc[
            instance_df["correct"],
            "top1_attribution_share",
        ]
    ),

    "incorrect_mean_top1_attribution_share": safe_mean(
        instance_df.loc[
            ~instance_df["correct"],
            "top1_attribution_share",
        ]
    ),

    "correct_mean_context_attribution_ratio": (
        safe_mean(
            instance_df.loc[
                instance_df["correct"],
                "context_attribution_ratio",
            ]
        )
    ),

    "incorrect_mean_context_attribution_ratio": (
        safe_mean(
            instance_df.loc[
                ~instance_df["correct"],
                "context_attribution_ratio",
            ]
        )
    ),

    "low_confidence_error_threshold": 0.50,

    "low_confidence_errors": (
        low_confidence_error_count
    ),

    "high_confidence_error_threshold": 0.80,

    "high_confidence_errors": (
        high_confidence_error_count
    ),

    "high_concentration_error_threshold": 0.30,

    "high_concentration_errors": (
        high_concentration_error_count
    ),

    "confidence_proxy_definition": (
        "Softmax-normalized distribution over summed "
        "class-specific SHAP values. This is an "
        "attribution-derived confidence proxy, not a "
        "calibrated model probability."
    ),

    "class_margin_definition": (
        "Predicted-class summed SHAP score minus the "
        "strongest competing-class summed SHAP score."
    ),
}


json_output = (
    OUTPUT_DIR
    / "confidence_attribution_analysis_summary.json"
)

with open(
    json_output,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_json,
        f,
        indent=4,
    )

print(
    json.dumps(
        summary_json,
        indent=4,
    )
)

print(
    f"\nSaved: {json_output}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE-ATTRIBUTION ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

generated_files = [
    "confidence_attribution_by_instance.csv",
    "confidence_attribution_summary.csv",
    "confidence_attribution_by_correctness.csv",
    "confidence_attribution_by_confidence_bin.csv",
    "confidence_attribution_by_error_type.csv",
    "low_confidence_errors.csv",
    "high_confidence_errors.csv",
    "high_confidence_high_concentration_errors.csv",
    "confidence_attribution_correlations.csv",
    "confidence_attribution_analysis_summary.json",
]

for filename in generated_files:
    print(f" - {filename}")
