from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("results/transformer/distilbert_baseline/explainability")

INPUT_FILE = (
    BASE_DIR
    / "aspect_context_analysis"
    / "aspect_context_by_instance.csv"
)

OUTPUT_DIR = BASE_DIR / "attribution_gap_analysis"


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
TOP_N_ERRORS = 50


# ============================================================
# HELPERS
# ============================================================

def safe_rate(numerator, denominator):
    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SHAP ANALYSIS — EXPERIMENT 4")
    print("ATTRIBUTION GAP / CONTEXT DOMINANCE vs ERROR")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print("Loading aspect/context attribution data...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows      : {len(df):,}")
    print(
        f"Instances : "
        f"{df[['sentence_id', 'aspect']].drop_duplicates().shape[0]:,}"
    )

    required_columns = [
        "sentence_id",
        "aspect",
        "sentence",
        "true_polarity",
        "predicted_polarity",
        "correct",
        "aspect_abs_shap",
        "context_abs_shap",
        "aspect_attribution_ratio",
        "context_attribution_ratio",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df["correct"] = df["correct"].astype(bool)

    numeric_columns = [
        "aspect_abs_shap",
        "context_abs_shap",
        "aspect_attribution_ratio",
        "context_attribution_ratio",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # The input should contain one row per instance.
    # If duplicate rows somehow exist, keep one row per
    # sentence_id + aspect.

    instance_columns = [
        "sentence_id",
        "aspect",
    ]

    duplicates = df.duplicated(instance_columns).sum()

    if duplicates > 0:
        print(
            f"WARNING: {duplicates:,} duplicate instance rows found. "
            "Keeping first occurrence."
        )

        df = df.drop_duplicates(
            subset=instance_columns,
            keep="first"
        ).copy()

    print(f"Unique instances: {len(df):,}")

    # --------------------------------------------------------
    # ATTRIBUTION GAP
    # --------------------------------------------------------

    print_header("CALCULATING ATTRIBUTION GAP")

    df["attribution_gap"] = (
        df["context_attribution_ratio"]
        - df["aspect_attribution_ratio"]
    )

    df["absolute_attribution_gap"] = (
        df["attribution_gap"].abs()
    )

    df["context_minus_aspect_abs_shap"] = (
        df["context_abs_shap"]
        - df["aspect_abs_shap"]
    )

    df["context_dominant"] = (
        df["context_abs_shap"]
        > df["aspect_abs_shap"]
    )

    df["aspect_dominant"] = (
        df["aspect_abs_shap"]
        > df["context_abs_shap"]
    )

    df["tie"] = (
        df["aspect_abs_shap"]
        == df["context_abs_shap"]
    )

    print(
        "Mean attribution gap:",
        f"{df['attribution_gap'].mean():.6f}"
    )

    print(
        "Median attribution gap:",
        f"{df['attribution_gap'].median():.6f}"
    )

    print(
        "Mean context attribution ratio:",
        f"{df['context_attribution_ratio'].mean():.6f}"
    )

    print(
        "Mean aspect attribution ratio:",
        f"{df['aspect_attribution_ratio'].mean():.6f}"
    )

    # --------------------------------------------------------
    # SAVE INSTANCE-LEVEL DATA
    # --------------------------------------------------------

    instance_output = (
        OUTPUT_DIR
        / "attribution_gap_by_instance.csv"
    )

    df.sort_values(
        "attribution_gap",
        ascending=False
    ).to_csv(
        instance_output,
        index=False
    )

    print(f"Saved: {instance_output}")

    # --------------------------------------------------------
    # CORRECT VS INCORRECT
    # --------------------------------------------------------

    print_header("CORRECT VS INCORRECT")

    summary_rows = []

    for correct_value, group in df.groupby("correct"):

        summary_rows.append(
            {
                "correct": correct_value,
                "instances": len(group),

                "mean_attribution_gap":
                    group["attribution_gap"].mean(),

                "median_attribution_gap":
                    group["attribution_gap"].median(),

                "std_attribution_gap":
                    group["attribution_gap"].std(),

                "mean_absolute_attribution_gap":
                    group["absolute_attribution_gap"].mean(),

                "median_absolute_attribution_gap":
                    group["absolute_attribution_gap"].median(),

                "mean_context_attribution_ratio":
                    group["context_attribution_ratio"].mean(),

                "median_context_attribution_ratio":
                    group["context_attribution_ratio"].median(),

                "mean_aspect_attribution_ratio":
                    group["aspect_attribution_ratio"].mean(),

                "median_aspect_attribution_ratio":
                    group["aspect_attribution_ratio"].median(),

                "context_dominant_instances":
                    int(group["context_dominant"].sum()),

                "aspect_dominant_instances":
                    int(group["aspect_dominant"].sum()),

                "tie_instances":
                    int(group["tie"].sum()),

                "context_dominant_rate":
                    group["context_dominant"].mean(),

                "aspect_dominant_rate":
                    group["aspect_dominant"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    print(
        summary_df.to_string(index=False)
    )

    summary_output = (
        OUTPUT_DIR
        / "attribution_gap_summary.csv"
    )

    summary_df.to_csv(
        summary_output,
        index=False
    )

    print(f"Saved: {summary_output}")

    # --------------------------------------------------------
    # ERROR TYPE ANALYSIS
    # --------------------------------------------------------

    print_header("ERROR TYPE ANALYSIS")

    errors = df[~df["correct"]].copy()

    error_type_rows = []

    for (
        true_label,
        predicted_label
    ), group in errors.groupby(
        ["true_polarity", "predicted_polarity"]
    ):

        error_type_rows.append(
            {
                "true_polarity": true_label,
                "predicted_polarity": predicted_label,
                "instances": len(group),

                "mean_attribution_gap":
                    group["attribution_gap"].mean(),

                "median_attribution_gap":
                    group["attribution_gap"].median(),

                "mean_absolute_attribution_gap":
                    group["absolute_attribution_gap"].mean(),

                "mean_context_attribution_ratio":
                    group["context_attribution_ratio"].mean(),

                "mean_aspect_attribution_ratio":
                    group["aspect_attribution_ratio"].mean(),

                "context_dominant_instances":
                    int(group["context_dominant"].sum()),

                "aspect_dominant_instances":
                    int(group["aspect_dominant"].sum()),

                "context_dominant_rate":
                    group["context_dominant"].mean(),

                "mean_context_minus_aspect_abs_shap":
                    group[
                        "context_minus_aspect_abs_shap"
                    ].mean(),
            }
        )

    error_type_df = pd.DataFrame(error_type_rows)

    if not error_type_df.empty:

        error_type_df = error_type_df.sort_values(
            "mean_attribution_gap",
            ascending=False
        )

        print(
            error_type_df.to_string(index=False)
        )

    error_type_output = (
        OUTPUT_DIR
        / "attribution_gap_by_error_type.csv"
    )

    error_type_df.to_csv(
        error_type_output,
        index=False
    )

    print(f"Saved: {error_type_output}")

    # --------------------------------------------------------
    # THRESHOLD ANALYSIS
    # --------------------------------------------------------

    print_header("CONTEXT DOMINANCE THRESHOLD ANALYSIS")

    threshold_rows = []

    total_correct = int(
        df["correct"].sum()
    )

    total_incorrect = int(
        (~df["correct"]).sum()
    )

    for threshold in THRESHOLDS:

        dominant = (
            df["context_attribution_ratio"]
            >= threshold
        )

        total_count = int(
            dominant.sum()
        )

        correct_count = int(
            (
                dominant
                & df["correct"]
            ).sum()
        )

        incorrect_count = int(
            (
                dominant
                & ~df["correct"]
            ).sum()
        )

        threshold_rows.append(
            {
                "threshold": threshold,

                "instances":
                    total_count,

                "instance_rate":
                    safe_rate(
                        total_count,
                        len(df)
                    ),

                "correct_instances":
                    correct_count,

                "incorrect_instances":
                    incorrect_count,

                "correct_rate_within_threshold":
                    safe_rate(
                        correct_count,
                        total_count
                    ),

                "error_rate_within_threshold":
                    safe_rate(
                        incorrect_count,
                        total_count
                    ),

                "fraction_of_all_correct":
                    safe_rate(
                        correct_count,
                        total_correct
                    ),

                "fraction_of_all_errors":
                    safe_rate(
                        incorrect_count,
                        total_incorrect
                    ),
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    print(
        threshold_df.to_string(index=False)
    )

    threshold_output = (
        OUTPUT_DIR
        / "context_dominance_thresholds.csv"
    )

    threshold_df.to_csv(
        threshold_output,
        index=False
    )

    print(f"Saved: {threshold_output}")

    # --------------------------------------------------------
    # HIGH CONTEXT-DOMINANT ERRORS
    # --------------------------------------------------------

    print_header("HIGH CONTEXT-DOMINANT ERRORS")

    high_context_errors = errors.copy()

    high_context_errors = high_context_errors.sort_values(
        [
            "context_attribution_ratio",
            "attribution_gap",
        ],
        ascending=False
    )

    selected_columns = [
        "sentence_id",
        "aspect",
        "sentence",
        "true_polarity",
        "predicted_polarity",
        "aspect_abs_shap",
        "context_abs_shap",
        "aspect_attribution_ratio",
        "context_attribution_ratio",
        "attribution_gap",
        "context_minus_aspect_abs_shap",
        "strongest_context_token",
        "strongest_context_shap",
        "strongest_aspect_token",
        "strongest_aspect_shap",
    ]

    selected_columns = [
        c
        for c in selected_columns
        if c in high_context_errors.columns
    ]

    high_context_errors = high_context_errors[
        selected_columns
    ].head(TOP_N_ERRORS)

    display_columns = [
        c
        for c in [
            "sentence_id",
            "aspect",
            "true_polarity",
            "predicted_polarity",
            "context_attribution_ratio",
            "attribution_gap",
            "strongest_context_token",
            "strongest_context_shap",
        ]
        if c in high_context_errors.columns
    ]

    print(
        high_context_errors[
            display_columns
        ].to_string(index=False)
    )

    high_context_output = (
        OUTPUT_DIR
        / "high_context_dominant_errors.csv"
    )

    high_context_errors.to_csv(
        high_context_output,
        index=False
    )

    print(f"Saved: {high_context_output}")

    # --------------------------------------------------------
    # ASPECT-SPECIFIC ANALYSIS
    # --------------------------------------------------------

    print_header("ASPECT-LEVEL ATTRIBUTION ANALYSIS")

    aspect_rows = []

    for aspect, group in df.groupby("aspect"):

        if len(group) < 2:
            continue

        error_group = group[
            ~group["correct"]
        ]

        aspect_rows.append(
            {
                "aspect": aspect,
                "instances": len(group),
                "errors": len(error_group),

                "error_rate":
                    safe_rate(
                        len(error_group),
                        len(group)
                    ),

                "mean_context_attribution_ratio":
                    group[
                        "context_attribution_ratio"
                    ].mean(),

                "mean_aspect_attribution_ratio":
                    group[
                        "aspect_attribution_ratio"
                    ].mean(),

                "mean_attribution_gap":
                    group[
                        "attribution_gap"
                    ].mean(),

                "context_dominant_rate":
                    group[
                        "context_dominant"
                    ].mean(),

                "error_context_dominant_rate":
                    (
                        error_group[
                            "context_dominant"
                        ].mean()
                        if len(error_group) > 0
                        else np.nan
                    ),
            }
        )

    aspect_df = pd.DataFrame(
        aspect_rows
    )

    if not aspect_df.empty:

        aspect_df = aspect_df.sort_values(
            [
                "error_rate",
                "mean_attribution_gap",
            ],
            ascending=False
        )

        print(
            aspect_df.head(30).to_string(
                index=False
            )
        )

    aspect_output = (
        OUTPUT_DIR
        / "aspect_attribution_gap.csv"
    )

    aspect_df.to_csv(
        aspect_output,
        index=False
    )

    print(f"Saved: {aspect_output}")

    # --------------------------------------------------------
    # SUMMARY JSON
    # --------------------------------------------------------

    print_header("CREATING SUMMARY")

    total_instances = len(df)

    correct_instances = int(
        df["correct"].sum()
    )

    incorrect_instances = int(
        (~df["correct"]).sum()
    )

    context_dominant_instances = int(
        df["context_dominant"].sum()
    )

    aspect_dominant_instances = int(
        df["aspect_dominant"].sum()
    )

    summary = {
        "total_instances":
            total_instances,

        "correct_instances":
            correct_instances,

        "incorrect_instances":
            incorrect_instances,

        "accuracy":
            safe_rate(
                correct_instances,
                total_instances
            ),

        "overall_mean_attribution_gap":
            float(
                df["attribution_gap"].mean()
            ),

        "overall_median_attribution_gap":
            float(
                df["attribution_gap"].median()
            ),

        "overall_mean_context_attribution_ratio":
            float(
                df[
                    "context_attribution_ratio"
                ].mean()
            ),

        "overall_mean_aspect_attribution_ratio":
            float(
                df[
                    "aspect_attribution_ratio"
                ].mean()
            ),

        "overall_context_dominant_instances":
            context_dominant_instances,

        "overall_aspect_dominant_instances":
            aspect_dominant_instances,

        "overall_context_dominant_rate":
            safe_rate(
                context_dominant_instances,
                total_instances
            ),

        "overall_aspect_dominant_rate":
            safe_rate(
                aspect_dominant_instances,
                total_instances
            ),

        "correct_mean_attribution_gap":
            float(
                df.loc[
                    df["correct"],
                    "attribution_gap"
                ].mean()
            ),

        "incorrect_mean_attribution_gap":
            float(
                df.loc[
                    ~df["correct"],
                    "attribution_gap"
                ].mean()
            ),

        "correct_mean_context_attribution_ratio":
            float(
                df.loc[
                    df["correct"],
                    "context_attribution_ratio"
                ].mean()
            ),

        "incorrect_mean_context_attribution_ratio":
            float(
                df.loc[
                    ~df["correct"],
                    "context_attribution_ratio"
                ].mean()
            ),

        "correct_context_dominant_rate":
            float(
                df.loc[
                    df["correct"],
                    "context_dominant"
                ].mean()
            ),

        "incorrect_context_dominant_rate":
            float(
                df.loc[
                    ~df["correct"],
                    "context_dominant"
                ].mean()
            ),
    }

    summary_output = (
        OUTPUT_DIR
        / "attribution_gap_analysis_summary.json"
    )

    with open(
        summary_output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        json.dumps(
            summary,
            indent=4
        )
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ATTRIBUTION GAP ANALYSIS COMPLETE")
    print("=" * 70)

    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("Generated files:")

    print(
        " - attribution_gap_by_instance.csv"
    )

    print(
        " - attribution_gap_summary.csv"
    )

    print(
        " - attribution_gap_by_error_type.csv"
    )

    print(
        " - context_dominance_thresholds.csv"
    )

    print(
        " - high_context_dominant_errors.csv"
    )

    print(
        " - aspect_attribution_gap.csv"
    )

    print(
        " - attribution_gap_analysis_summary.json"
    )


if __name__ == "__main__":
    main()
