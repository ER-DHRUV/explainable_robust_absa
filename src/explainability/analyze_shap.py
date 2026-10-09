from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 20

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPLAINABILITY_DIR = (
    PROJECT_ROOT
    / "results"
    / "transformer"
    / "distilbert_baseline"
    / "explainability"
)

INPUT_FILE = EXPLAINABILITY_DIR / "explanation_results.csv"

OUTPUT_DIR = EXPLAINABILITY_DIR / "shap_analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def clean_token(token: str) -> str:
    """
    Clean SHAP token strings while preserving the actual
    lexical content as much as possible.
    """
    if pd.isna(token):
        return ""

    token = str(token)

    # SHAP output can contain trailing whitespace.
    token = token.strip()

    # WordPiece continuation marker.
    token = token.replace("##", "")

    return token


def normalize_word(token: str) -> str:
    """
    Normalize a token for matching against an aspect term.
    """
    token = clean_token(token)

    token = re.sub(
        r"[^A-Za-z0-9'-]+",
        "",
        token,
    )

    return token.lower()


def aspect_words(aspect: str) -> list[str]:
    """
    Split an aspect into normalized words.
    """
    if pd.isna(aspect):
        return []

    return [
        normalize_word(x)
        for x in str(aspect).split()
        if normalize_word(x)
    ]


def is_aspect_token(
    token: str,
    aspect: str,
) -> bool:
    """
    Approximate whether a SHAP token belongs to the
    aspect term.

    This works at the token level and therefore deliberately
    uses substring matching to handle WordPiece fragments.
    """
    token_norm = normalize_word(token)

    if not token_norm:
        return False

    words = aspect_words(aspect)

    if not words:
        return False

    return any(
        token_norm in word
        or word in token_norm
        for word in words
    )


def load_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"SHAP results not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
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
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    df["correct"] = df["correct"].astype(bool)

    df["token"] = df["token"].fillna("")

    df["token_clean"] = df["token"].map(
        clean_token
    )

    df["token_normalized"] = df["token"].map(
        normalize_word
    )

    df["abs_shap"] = (
        df["shap_predicted_class"]
        .abs()
    )

    df["aspect_token"] = df.apply(
        lambda row: is_aspect_token(
            row["token"],
            row["aspect"],
        ),
        axis=1,
    )

    return df


# ============================================================
# 1. TOKEN IMPORTANCE
# ============================================================

def analyze_token_importance(
    df: pd.DataFrame,
):

    grouped = (
        df.groupby(
            [
                "token_normalized",
                "token_clean",
            ],
            dropna=False,
        )
        .agg(
            instances=(
                "shap_predicted_class",
                "count",
            ),
            mean_shap=(
                "shap_predicted_class",
                "mean",
            ),
            mean_abs_shap=(
                "abs_shap",
                "mean",
            ),
            max_abs_shap=(
                "abs_shap",
                "max",
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["token_normalized"] != ""
    ]

    grouped = grouped.sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    grouped.to_csv(
        OUTPUT_DIR / "token_importance.csv",
        index=False,
    )

    return grouped


# ============================================================
# 2. CORRECT VS INCORRECT
# ============================================================

def analyze_correct_vs_incorrect(
    df: pd.DataFrame,
):

    result = (
        df.groupby("correct")
        .agg(
            token_count=(
                "shap_predicted_class",
                "count",
            ),
            mean_shap=(
                "shap_predicted_class",
                "mean",
            ),
            median_shap=(
                "shap_predicted_class",
                "median",
            ),
            std_shap=(
                "shap_predicted_class",
                "std",
            ),
            mean_abs_shap=(
                "abs_shap",
                "mean",
            ),
            median_abs_shap=(
                "abs_shap",
                "median",
            ),
            max_abs_shap=(
                "abs_shap",
                "max",
            ),
        )
        .reset_index()
    )

    result.to_csv(
        OUTPUT_DIR / "correct_vs_incorrect.csv",
        index=False,
    )

    return result


# ============================================================
# 3. ERROR TYPE ANALYSIS
# ============================================================

def analyze_error_types(
    df: pd.DataFrame,
):

    errors = df[
        ~df["correct"]
    ].copy()

    result = (
        errors.groupby(
            [
                "true_polarity",
                "predicted_polarity",
            ]
        )
        .agg(
            token_count=(
                "shap_predicted_class",
                "count",
            ),
            mean_shap=(
                "shap_predicted_class",
                "mean",
            ),
            mean_abs_shap=(
                "abs_shap",
                "mean",
            ),
            max_abs_shap=(
                "abs_shap",
                "max",
            ),
        )
        .reset_index()
    )

    result["estimated_instances"] = (
        result["token_count"]
        / df.groupby(
            ["sentence_id", "aspect"]
        ).ngroups
    )

    result.to_csv(
        OUTPUT_DIR / "error_type_analysis.csv",
        index=False,
    )

    return result


# ============================================================
# 4. ASPECT ATTRIBUTION
# ============================================================

def analyze_aspect_attribution(
    df: pd.DataFrame,
):

    grouped = (
        df.groupby(
            [
                "sentence_id",
                "aspect",
            ]
        )
        .agg(
            true_polarity=(
                "true_polarity",
                "first",
            ),
            predicted_polarity=(
                "predicted_polarity",
                "first",
            ),
            correct=(
                "correct",
                "first",
            ),
            total_tokens=(
                "token",
                "count",
            ),
            aspect_tokens=(
                "aspect_token",
                "sum",
            ),
        )
        .reset_index()
    )

    # Calculate absolute SHAP separately for aspect and context.
    aspect_df = df[
        df["aspect_token"]
    ]

    context_df = df[
        ~df["aspect_token"]
    ]

    aspect_stats = (
        aspect_df.groupby(
            [
                "sentence_id",
                "aspect",
            ]
        )["abs_shap"]
        .agg(
            aspect_mean_abs_shap="mean",
            aspect_max_abs_shap="max",
        )
        .reset_index()
    )

    context_stats = (
        context_df.groupby(
            [
                "sentence_id",
                "aspect",
            ]
        )["abs_shap"]
        .agg(
            context_mean_abs_shap="mean",
            context_max_abs_shap="max",
        )
        .reset_index()
    )

    result = grouped.merge(
        aspect_stats,
        on=[
            "sentence_id",
            "aspect",
        ],
        how="left",
    )

    result = result.merge(
        context_stats,
        on=[
            "sentence_id",
            "aspect",
        ],
        how="left",
    )

    result["aspect_mean_abs_shap"] = (
        result["aspect_mean_abs_shap"]
        .fillna(0)
    )

    result["context_mean_abs_shap"] = (
        result["context_mean_abs_shap"]
        .fillna(0)
    )

    result["aspect_to_context_ratio"] = (
        result["aspect_mean_abs_shap"]
        / (
            result["context_mean_abs_shap"]
            + 1e-12
        )
    )

    result.to_csv(
        OUTPUT_DIR / "aspect_attribution.csv",
        index=False,
    )

    return result


# ============================================================
# 5. CONTEXT ATTRIBUTION
# ============================================================

def analyze_context_attribution(
    df: pd.DataFrame,
):

    result = (
        df[~df["aspect_token"]]
        .groupby(
            [
                "sentence_id",
                "aspect",
                "true_polarity",
                "predicted_polarity",
                "correct",
                "token_clean",
            ]
        )
        .agg(
            mean_abs_shap=(
                "abs_shap",
                "mean",
            ),
            mean_shap=(
                "shap_predicted_class",
                "mean",
            ),
            occurrences=(
                "token",
                "count",
            ),
        )
        .reset_index()
    )

    result = result.sort_values(
        [
            "sentence_id",
            "aspect",
            "mean_abs_shap",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    result.to_csv(
        OUTPUT_DIR / "context_attribution.csv",
        index=False,
    )

    return result


# ============================================================
# 6. MISLEADING EVIDENCE
# ============================================================

def analyze_misleading_evidence(
    df: pd.DataFrame,
):

    errors = df[
        ~df["correct"]
    ].copy()

    errors = errors[
        errors["token_normalized"] != ""
    ]

    result = errors[
        [
            "sentence_id",
            "aspect",
            "sentence",
            "true_polarity",
            "predicted_polarity",
            "token_index",
            "token_clean",
            "shap_predicted_class",
            "abs_shap",
            "aspect_token",
        ]
    ].copy()

    result = result.sort_values(
        "abs_shap",
        ascending=False,
    )

    result.to_csv(
        OUTPUT_DIR / "misleading_evidence.csv",
        index=False,
    )

    return result


# ============================================================
# 7. TOP ERROR EXPLANATIONS
# ============================================================

def analyze_top_error_explanations(
    df: pd.DataFrame,
):

    errors = df[
        ~df["correct"]
    ].copy()

    errors = errors[
        errors["token_normalized"] != ""
    ]

    rows = []

    for (
        sentence_id,
        aspect,
    ), group in errors.groupby(
        [
            "sentence_id",
            "aspect",
        ]
    ):

        group = group.sort_values(
            "abs_shap",
            ascending=False,
        )

        first = group.iloc[0]

        top_tokens = []

        for _, row in group.head(TOP_K).iterrows():

            top_tokens.append(
                {
                    "token": row["token_clean"],
                    "shap": float(
                        row[
                            "shap_predicted_class"
                        ]
                    ),
                    "abs_shap": float(
                        row["abs_shap"]
                    ),
                    "aspect_token": bool(
                        row["aspect_token"]
                    ),
                }
            )

        rows.append(
            {
                "sentence_id": sentence_id,
                "aspect": aspect,
                "sentence": first["sentence"],
                "true_polarity": first[
                    "true_polarity"
                ],
                "predicted_polarity": first[
                    "predicted_polarity"
                ],
                "strongest_token": first[
                    "token_clean"
                ],
                "strongest_shap": float(
                    first[
                        "shap_predicted_class"
                    ]
                ),
                "strongest_abs_shap": float(
                    first["abs_shap"]
                ),
                "top_tokens": json.dumps(
                    top_tokens,
                    ensure_ascii=False,
                ),
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "strongest_abs_shap",
        ascending=False,
    )

    result.to_csv(
        OUTPUT_DIR
        / "top_error_explanations.csv",
        index=False,
    )

    return result


# ============================================================
# 8. SUMMARY
# ============================================================

def create_summary(
    df: pd.DataFrame,
    aspect_df: pd.DataFrame,
    misleading_df: pd.DataFrame,
):

    instances = (
        df[
            [
                "sentence_id",
                "aspect",
            ]
        ]
        .drop_duplicates()
    )

    correct_instances = (
        df[
            [
                "sentence_id",
                "aspect",
                "correct",
            ]
        ]
        .drop_duplicates()
    )

    total_instances = len(instances)

    correct_count = int(
        correct_instances["correct"].sum()
    )

    incorrect_count = (
        total_instances
        - correct_count
    )

    summary = {
        "total_instances": total_instances,
        "correct_instances": correct_count,
        "incorrect_instances": incorrect_count,
        "accuracy": (
            correct_count
            / total_instances
            if total_instances
            else 0.0
        ),
        "total_tokens": int(len(df)),
        "mean_abs_shap": float(
            df["abs_shap"].mean()
        ),
        "median_abs_shap": float(
            df["abs_shap"].median()
        ),
        "correct_mean_abs_shap": float(
            df.loc[
                df["correct"],
                "abs_shap",
            ].mean()
        ),
        "incorrect_mean_abs_shap": float(
            df.loc[
                ~df["correct"],
                "abs_shap",
            ].mean()
        ),
        "aspect_token_rows": int(
            df["aspect_token"].sum()
        ),
        "context_token_rows": int(
            (~df["aspect_token"]).sum()
        ),
        "strongest_misleading_token": (
            misleading_df.iloc[0][
                "token_clean"
            ]
            if len(misleading_df)
            else None
        ),
        "strongest_misleading_shap": (
            float(
                misleading_df.iloc[0][
                    "shap_predicted_class"
                ]
            )
            if len(misleading_df)
            else None
        ),
    }

    with open(
        OUTPUT_DIR
        / "shap_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SHAP ANALYSIS — EXPERIMENT 2")
    print("=" * 70)

    print()
    print("Loading SHAP explanations...")

    df = load_data()

    print(
        f"Rows            : {len(df):,}"
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
        f"Instances       : {instance_count:,}"
    )

    print(
        f"Correct         : "
        f"{df.groupby(['sentence_id', 'aspect'])['correct'].first().sum():,}"
    )

    print(
        f"Incorrect       : "
        f"{instance_count - df.groupby(['sentence_id', 'aspect'])['correct'].first().sum():,}"
    )

    print()
    print("=" * 70)
    print("ANALYZING TOKEN IMPORTANCE")
    print("=" * 70)

    token_importance = (
        analyze_token_importance(df)
    )

    print(
        "Saved: token_importance.csv"
    )

    print()
    print("=" * 70)
    print("CORRECT VS INCORRECT")
    print("=" * 70)

    correct_vs_incorrect = (
        analyze_correct_vs_incorrect(df)
    )

    print(
        correct_vs_incorrect.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("ERROR TYPE ANALYSIS")
    print("=" * 70)

    error_types = analyze_error_types(df)

    print(
        error_types.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("ASPECT ATTRIBUTION")
    print("=" * 70)

    aspect_df = analyze_aspect_attribution(
        df
    )

    print(
        "Saved: aspect_attribution.csv"
    )

    print()
    print("=" * 70)
    print("CONTEXT ATTRIBUTION")
    print("=" * 70)

    analyze_context_attribution(df)

    print(
        "Saved: context_attribution.csv"
    )

    print()
    print("=" * 70)
    print("MISLEADING EVIDENCE")
    print("=" * 70)

    misleading_df = (
        analyze_misleading_evidence(df)
    )

    print(
        misleading_df[
            [
                "sentence_id",
                "aspect",
                "true_polarity",
                "predicted_polarity",
                "token_clean",
                "shap_predicted_class",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(
        "Saved: misleading_evidence.csv"
    )

    print()
    print("=" * 70)
    print("TOP ERROR EXPLANATIONS")
    print("=" * 70)

    top_errors = (
        analyze_top_error_explanations(df)
    )

    print(
        top_errors[
            [
                "sentence_id",
                "aspect",
                "true_polarity",
                "predicted_polarity",
                "strongest_token",
                "strongest_shap",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(
        "Saved: top_error_explanations.csv"
    )

    print()
    print("=" * 70)
    print("CREATING SUMMARY")
    print("=" * 70)

    summary = create_summary(
        df,
        aspect_df,
        misleading_df,
    )

    print(
        json.dumps(
            summary,
            indent=4,
        )
    )

    print()
    print("=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    print()
    print("Generated files:")

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):

        if path.is_file():

            print(
                f"  - {path.name}"
            )


if __name__ == "__main__":
    main()