from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ROBUSTNESS EXPERIMENT 4
# ROBUSTNESS ERROR ANALYSIS
# ============================================================

SEED = 42

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROBUSTNESS_DIR = (
    PROJECT_ROOT
    / "results"
    / "transformer"
    / "distilbert_baseline"
    / "robustness"
)

BASELINE_DIR = (
    ROBUSTNESS_DIR
    / "baseline_evaluation"
)

OUTPUT_DIR = (
    ROBUSTNESS_DIR
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

BASELINE_RESULTS = (
    BASELINE_DIR
    / "baseline_perturbation_results.csv"
)

BY_PERTURBATION = (
    BASELINE_DIR
    / "robustness_by_perturbation_type.csv"
)

BY_POLARITY = (
    BASELINE_DIR
    / "robustness_by_polarity.csv"
)

FLIP_ANALYSIS = (
    BASELINE_DIR
    / "prediction_flip_analysis.csv"
)

HIGH_CONFIDENCE_FLIPS = (
    BASELINE_DIR
    / "high_confidence_prediction_flips.csv"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def safe_float(value) -> float:

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# LOAD DATA
# ============================================================

def load_inputs():

    print("=" * 70)
    print("ROBUSTNESS EXPERIMENT 4")
    print("ROBUSTNESS ERROR ANALYSIS")
    print("=" * 70)

    print()
    print("Loading Experiment 2 outputs...")

    required_files = [
        BASELINE_RESULTS,
        BY_PERTURBATION,
        BY_POLARITY,
        FLIP_ANALYSIS,
        HIGH_CONFIDENCE_FLIPS,
    ]

    for path in required_files:
        require_file(path)

    results_df = pd.read_csv(
        BASELINE_RESULTS
    )

    perturbation_df = pd.read_csv(
        BY_PERTURBATION
    )

    polarity_df = pd.read_csv(
        BY_POLARITY
    )

    flip_df = pd.read_csv(
        FLIP_ANALYSIS
    )

    high_conf_df = pd.read_csv(
        HIGH_CONFIDENCE_FLIPS
    )

    print(
        f"Baseline perturbation rows : "
        f"{len(results_df):,}"
    )

    print(
        f"Perturbation types          : "
        f"{len(perturbation_df):,}"
    )

    print(
        f"Polarity groups             : "
        f"{len(polarity_df):,}"
    )

    print(
        f"Prediction flip categories  : "
        f"{len(flip_df):,}"
    )

    print(
        f"High-confidence failures    : "
        f"{len(high_conf_df):,}"
    )

    return (
        results_df,
        perturbation_df,
        polarity_df,
        flip_df,
        high_conf_df,
    )


# ============================================================
# VALIDATE BASELINE RESULTS
# ============================================================

def validate_columns(df: pd.DataFrame):

    required = {
        "perturbation_id",
        "sentence_id",
        "aspect",
        "true_polarity",
        "perturbation_type",
        "original_predicted_polarity",
        "perturbed_predicted_polarity",
        "original_confidence",
        "perturbed_confidence",
        "confidence_change",
        "original_correct",
        "perturbed_correct",
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            "baseline_perturbation_results.csv "
            f"is missing columns: {sorted(missing)}"
        )


# ============================================================
# ERROR FLAGS
# ============================================================

def create_error_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["prediction_flip"] = (
        df["original_predicted_polarity"]
        != df["perturbed_predicted_polarity"]
    )

    df["correct_to_incorrect"] = (
        (df["original_correct"] == True)
        & (df["perturbed_correct"] == False)
    )

    df["incorrect_to_correct"] = (
        (df["original_correct"] == False)
        & (df["perturbed_correct"] == True)
    )

    df["remained_correct"] = (
        (df["original_correct"] == True)
        & (df["perturbed_correct"] == True)
    )

    df["remained_incorrect"] = (
        (df["original_correct"] == False)
        & (df["perturbed_correct"] == False)
    )

    df["absolute_confidence_change"] = (
        df["confidence_change"]
        .abs()
    )

    return df


# ============================================================
# OVERALL ERROR ANALYSIS
# ============================================================

def overall_error_analysis(
    df: pd.DataFrame,
):

    total = len(df)

    flips = int(
        df["prediction_flip"].sum()
    )

    correct_to_incorrect = int(
        df["correct_to_incorrect"].sum()
    )

    incorrect_to_correct = int(
        df["incorrect_to_correct"].sum()
    )

    remained_correct = int(
        df["remained_correct"].sum()
    )

    remained_incorrect = int(
        df["remained_incorrect"].sum()
    )

    analysis = {
        "total_perturbed_instances": total,

        "prediction_flips": flips,

        "prediction_flip_rate": (
            flips / total
            if total
            else 0.0
        ),

        "correct_to_incorrect": (
            correct_to_incorrect
        ),

        "correct_to_incorrect_rate": (
            correct_to_incorrect / total
            if total
            else 0.0
        ),

        "incorrect_to_correct": (
            incorrect_to_correct
        ),

        "incorrect_to_correct_rate": (
            incorrect_to_correct / total
            if total
            else 0.0
        ),

        "remained_correct": (
            remained_correct
        ),

        "remained_incorrect": (
            remained_incorrect
        ),

        "mean_confidence_change": safe_float(
            df["confidence_change"].mean()
        ),

        "mean_absolute_confidence_change": safe_float(
            df[
                "absolute_confidence_change"
            ].mean()
        ),
    }

    return analysis


# ============================================================
# ERROR ANALYSIS BY PERTURBATION TYPE
# ============================================================

def analyze_perturbation_types(
    df: pd.DataFrame,
):

    grouped = []

    for perturbation_type, group in (
        df.groupby("perturbation_type")
    ):

        total = len(group)

        flips = int(
            group["prediction_flip"].sum()
        )

        c2i = int(
            group["correct_to_incorrect"].sum()
        )

        i2c = int(
            group["incorrect_to_correct"].sum()
        )

        grouped.append(
            {
                "perturbation_type":
                    perturbation_type,

                "instances":
                    total,

                "prediction_flips":
                    flips,

                "prediction_flip_rate":
                    flips / total
                    if total else 0.0,

                "correct_to_incorrect":
                    c2i,

                "correct_to_incorrect_rate":
                    c2i / total
                    if total else 0.0,

                "incorrect_to_correct":
                    i2c,

                "incorrect_to_correct_rate":
                    i2c / total
                    if total else 0.0,

                "mean_confidence_change":
                    safe_float(
                        group[
                            "confidence_change"
                        ].mean()
                    ),

                "mean_absolute_confidence_change":
                    safe_float(
                        group[
                            "absolute_confidence_change"
                        ].mean()
                    ),

                "mean_original_confidence":
                    safe_float(
                        group[
                            "original_confidence"
                        ].mean()
                    ),

                "mean_perturbed_confidence":
                    safe_float(
                        group[
                            "perturbed_confidence"
                        ].mean()
                    ),
            }
        )

    result = pd.DataFrame(grouped)

    if not result.empty:

        result = result.sort_values(
            "correct_to_incorrect_rate",
            ascending=False,
        )

    return result


# ============================================================
# ERROR ANALYSIS BY POLARITY
# ============================================================

def analyze_polarity(
    df: pd.DataFrame,
):

    grouped = []

    for polarity, group in (
        df.groupby("true_polarity")
    ):

        total = len(group)

        flips = int(
            group["prediction_flip"].sum()
        )

        c2i = int(
            group["correct_to_incorrect"].sum()
        )

        i2c = int(
            group["incorrect_to_correct"].sum()
        )

        grouped.append(
            {
                "true_polarity":
                    polarity,

                "instances":
                    total,

                "prediction_flips":
                    flips,

                "prediction_flip_rate":
                    flips / total
                    if total else 0.0,

                "correct_to_incorrect":
                    c2i,

                "correct_to_incorrect_rate":
                    c2i / total
                    if total else 0.0,

                "incorrect_to_correct":
                    i2c,

                "incorrect_to_correct_rate":
                    i2c / total
                    if total else 0.0,

                "mean_confidence_change":
                    safe_float(
                        group[
                            "confidence_change"
                        ].mean()
                    ),

                "mean_absolute_confidence_change":
                    safe_float(
                        group[
                            "absolute_confidence_change"
                        ].mean()
                    ),
            }
        )

    result = pd.DataFrame(grouped)

    return result


# ============================================================
# ERROR ANALYSIS BY ASPECT
# ============================================================

def analyze_aspects(
    df: pd.DataFrame,
):

    grouped = []

    for aspect, group in (
        df.groupby("aspect")
    ):

        total = len(group)

        flips = int(
            group["prediction_flip"].sum()
        )

        c2i = int(
            group["correct_to_incorrect"].sum()
        )

        i2c = int(
            group["incorrect_to_correct"].sum()
        )

        grouped.append(
            {
                "aspect":
                    aspect,

                "instances":
                    total,

                "prediction_flips":
                    flips,

                "prediction_flip_rate":
                    flips / total
                    if total else 0.0,

                "correct_to_incorrect":
                    c2i,

                "correct_to_incorrect_rate":
                    c2i / total
                    if total else 0.0,

                "incorrect_to_correct":
                    i2c,

                "incorrect_to_correct_rate":
                    i2c / total
                    if total else 0.0,

                "mean_confidence_change":
                    safe_float(
                        group[
                            "confidence_change"
                        ].mean()
                    ),

                "mean_absolute_confidence_change":
                    safe_float(
                        group[
                            "absolute_confidence_change"
                        ].mean()
                    ),
            }
        )

    result = pd.DataFrame(grouped)

    if not result.empty:

        result = result.sort_values(
            [
                "correct_to_incorrect_rate",
                "correct_to_incorrect",
            ],
            ascending=False,
        )

    return result


# ============================================================
# PREDICTION FLIP PATTERNS
# ============================================================

def analyze_flip_patterns(
    df: pd.DataFrame,
):

    flips = df[
        df["prediction_flip"]
    ].copy()

    if flips.empty:

        return pd.DataFrame(
            columns=[
                "original_prediction",
                "perturbed_prediction",
                "instances",
                "mean_confidence_change",
                "mean_absolute_confidence_change",
            ]
        )

    result = (
        flips
        .groupby(
            [
                "original_predicted_polarity",
                "perturbed_predicted_polarity",
            ]
        )
        .agg(
            instances=(
                "perturbation_id",
                "count",
            ),
            mean_confidence_change=(
                "confidence_change",
                "mean",
            ),
            mean_absolute_confidence_change=(
                "absolute_confidence_change",
                "mean",
            ),
        )
        .reset_index()
    )

    result = result.rename(
        columns={
            "original_predicted_polarity":
                "original_prediction",

            "perturbed_predicted_polarity":
                "perturbed_prediction",
        }
    )

    result = result.sort_values(
        "instances",
        ascending=False,
    )

    return result


# ============================================================
# HIGH-CONFIDENCE FAILURE ANALYSIS
# ============================================================

def high_confidence_failures(
    df: pd.DataFrame,
    threshold: float = 0.80,
):

    failures = df[
        (df["correct_to_incorrect"])
        & (
            df["original_confidence"]
            >= threshold
        )
    ].copy()

    if failures.empty:

        return failures

    failures = failures.sort_values(
        [
            "original_confidence",
            "absolute_confidence_change",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return failures


# ============================================================
# REPRESENTATIVE FAILURE CASES
# ============================================================

def representative_failure_cases(
    df: pd.DataFrame,
    n_per_type: int = 10,
):

    failures = df[
        df["correct_to_incorrect"]
    ].copy()

    if failures.empty:

        return failures

    failures = failures.sort_values(
        "absolute_confidence_change",
        ascending=False,
    )

    selected = []

    for perturbation_type, group in (
        failures.groupby(
            "perturbation_type"
        )
    ):

        selected.append(
            group.head(n_per_type)
        )

    if not selected:

        return pd.DataFrame()

    result = pd.concat(
        selected,
        ignore_index=True,
    )

    columns = [
        "perturbation_id",
        "sentence_id",
        "aspect",
        "true_polarity",
        "perturbation_type",
        "original_predicted_polarity",
        "perturbed_predicted_polarity",
        "original_confidence",
        "perturbed_confidence",
        "confidence_change",
        "original_sentence",
        "perturbed_sentence",
    ]

    available = [
        column
        for column in columns
        if column in result.columns
    ]

    return result[available]


# ============================================================
# ERROR CATEGORIES
# ============================================================

def classify_error_patterns(
    df: pd.DataFrame,
):

    failures = df[
        df["correct_to_incorrect"]
    ].copy()

    if failures.empty:

        return pd.DataFrame()

    failures["error_pattern"] = (
        failures.apply(
            classify_single_error,
            axis=1,
        )
    )

    result = (
        failures
        .groupby("error_pattern")
        .agg(
            instances=(
                "perturbation_id",
                "count",
            ),
            mean_confidence_change=(
                "confidence_change",
                "mean",
            ),
            mean_absolute_confidence_change=(
                "absolute_confidence_change",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "instances",
            ascending=False,
        )
    )

    return result


def classify_single_error(row):

    perturbation_type = (
        str(row["perturbation_type"])
        .lower()
    )

    original_prediction = (
        str(
            row[
                "original_predicted_polarity"
            ]
        ).lower()
    )

    perturbed_prediction = (
        str(
            row[
                "perturbed_predicted_polarity"
            ]
        ).lower()
    )

    if "context" in perturbation_type:

        return "context_sensitivity"

    if "noise" in perturbation_type:

        return "irrelevant_noise_sensitivity"

    if "punctuation" in perturbation_type:

        return "punctuation_sensitivity"

    if "whitespace" in perturbation_type:

        return "formatting_sensitivity"

    if "suffix" in perturbation_type:

        return "context_suffix_sensitivity"

    if (
        original_prediction
        != perturbed_prediction
    ):

        return "prediction_boundary_shift"

    return "other"


# ============================================================
# SAVE JSON
# ============================================================

def make_json_serializable(obj):

    if isinstance(
        obj,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):

        return int(obj)

    if isinstance(
        obj,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):

        return float(obj)

    if isinstance(obj, dict):

        return {
            key: make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):

        return [
            make_json_serializable(value)
            for value in obj
        ]

    return obj


# ============================================================
# MAIN
# ============================================================

def main():

    (
        results_df,
        perturbation_df,
        polarity_df,
        flip_df,
        high_conf_df,
    ) = load_inputs()

    validate_columns(
        results_df
    )

    print()
    print("=" * 70)
    print("CREATING ERROR FLAGS")
    print("=" * 70)

    results_df = create_error_flags(
        results_df
    )

    print(
        f"Prediction flips: "
        f"{results_df['prediction_flip'].sum():,}"
    )

    print(
        f"Correct → incorrect: "
        f"{results_df['correct_to_incorrect'].sum():,}"
    )

    print(
        f"Incorrect → correct: "
        f"{results_df['incorrect_to_correct'].sum():,}"
    )

    # --------------------------------------------------------
    # Overall analysis
    # --------------------------------------------------------

    overall = overall_error_analysis(
        results_df
    )

    print()
    print("=" * 70)
    print("OVERALL ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"Total perturbations       : "
        f"{overall['total_perturbed_instances']:,}"
    )

    print(
        f"Prediction flips          : "
        f"{overall['prediction_flips']:,}"
    )

    print(
        f"Prediction flip rate      : "
        f"{overall['prediction_flip_rate']:.4f}"
    )

    print(
        f"Correct → incorrect       : "
        f"{overall['correct_to_incorrect']:,}"
    )

    print(
        f"Correct → incorrect rate  : "
        f"{overall['correct_to_incorrect_rate']:.4f}"
    )

    print(
        f"Incorrect → correct       : "
        f"{overall['incorrect_to_correct']:,}"
    )

    print(
        f"Incorrect → correct rate  : "
        f"{overall['incorrect_to_correct_rate']:.4f}"
    )

    print(
        f"Mean confidence change   : "
        f"{overall['mean_confidence_change']:.6f}"
    )

    print(
        f"Mean absolute confidence Δ: "
        f"{overall['mean_absolute_confidence_change']:.6f}"
    )

    # --------------------------------------------------------
    # Perturbation analysis
    # --------------------------------------------------------

    perturbation_analysis = (
        analyze_perturbation_types(
            results_df
        )
    )

    print()
    print("=" * 70)
    print("ERROR ANALYSIS BY PERTURBATION TYPE")
    print("=" * 70)

    print(
        perturbation_analysis.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Polarity analysis
    # --------------------------------------------------------

    polarity_analysis = analyze_polarity(
        results_df
    )

    print()
    print("=" * 70)
    print("ERROR ANALYSIS BY TRUE POLARITY")
    print("=" * 70)

    print(
        polarity_analysis.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Aspect analysis
    # --------------------------------------------------------

    aspect_analysis = analyze_aspects(
        results_df
    )

    print()
    print("=" * 70)
    print("ERROR ANALYSIS BY ASPECT")
    print("=" * 70)

    print(
        aspect_analysis.head(30).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Flip patterns
    # --------------------------------------------------------

    flip_patterns = analyze_flip_patterns(
        results_df
    )

    print()
    print("=" * 70)
    print("PREDICTION FLIP PATTERNS")
    print("=" * 70)

    print(
        flip_patterns.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Error categories
    # --------------------------------------------------------

    error_patterns = (
        classify_error_patterns(
            results_df
        )
    )

    print()
    print("=" * 70)
    print("ERROR PATTERN CATEGORIES")
    print("=" * 70)

    if not error_patterns.empty:

        print(
            error_patterns.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # High-confidence failures
    # --------------------------------------------------------

    high_confidence = (
        high_confidence_failures(
            results_df,
            threshold=0.80,
        )
    )

    print()
    print("=" * 70)
    print("HIGH-CONFIDENCE ROBUSTNESS FAILURES")
    print("=" * 70)

    print(
        f"Failures with original confidence >= 0.80: "
        f"{len(high_confidence):,}"
    )

    if not high_confidence.empty:

        print(
            high_confidence[
                [
                    "perturbation_id",
                    "sentence_id",
                    "aspect",
                    "true_polarity",
                    "perturbation_type",
                    "original_predicted_polarity",
                    "perturbed_predicted_polarity",
                    "original_confidence",
                    "perturbed_confidence",
                    "confidence_change",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Representative failures
    # --------------------------------------------------------

    representative = (
        representative_failure_cases(
            results_df,
            n_per_type=10,
        )
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVING ERROR ANALYSIS")
    print("=" * 70)

    results_df.to_csv(
        OUTPUT_DIR
        / "robustness_error_analysis.csv",
        index=False,
    )

    perturbation_analysis.to_csv(
        OUTPUT_DIR
        / "error_analysis_by_perturbation.csv",
        index=False,
    )

    polarity_analysis.to_csv(
        OUTPUT_DIR
        / "error_analysis_by_polarity.csv",
        index=False,
    )

    aspect_analysis.to_csv(
        OUTPUT_DIR
        / "error_analysis_by_aspect.csv",
        index=False,
    )

    flip_patterns.to_csv(
        OUTPUT_DIR
        / "prediction_flip_patterns.csv",
        index=False,
    )

    error_patterns.to_csv(
        OUTPUT_DIR
        / "error_pattern_categories.csv",
        index=False,
    )

    high_confidence.to_csv(
        OUTPUT_DIR
        / "high_confidence_failures.csv",
        index=False,
    )

    representative.to_csv(
        OUTPUT_DIR
        / "representative_failure_cases.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment": "Robustness Experiment 4",

        "name": "Robustness Error Analysis",

        "source_experiment": (
            "Experiment 2 - "
            "Baseline Perturbation Evaluation"
        ),

        "total_perturbations": (
            int(len(results_df))
        ),

        "prediction_flips": (
            int(
                results_df[
                    "prediction_flip"
                ].sum()
            )
        ),

        "prediction_flip_rate": (
            safe_float(
                results_df[
                    "prediction_flip"
                ].mean()
            )
        ),

        "correct_to_incorrect": (
            int(
                results_df[
                    "correct_to_incorrect"
                ].sum()
            )
        ),

        "correct_to_incorrect_rate": (
            safe_float(
                results_df[
                    "correct_to_incorrect"
                ].mean()
            )
        ),

        "incorrect_to_correct": (
            int(
                results_df[
                    "incorrect_to_correct"
                ].sum()
            )
        ),

        "incorrect_to_correct_rate": (
            safe_float(
                results_df[
                    "incorrect_to_correct"
                ].mean()
            )
        ),

        "high_confidence_failure_threshold": 0.80,

        "high_confidence_failures": (
            int(len(high_confidence))
        ),

        "mean_confidence_change": (
            safe_float(
                results_df[
                    "confidence_change"
                ].mean()
            )
        ),

        "mean_absolute_confidence_change": (
            safe_float(
                results_df[
                    "absolute_confidence_change"
                ].mean()
            )
        ),

        "output_directory": str(
            OUTPUT_DIR
        ),
    }

    with open(
        OUTPUT_DIR
        / "robustness_error_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            make_json_serializable(
                summary
            ),
            f,
            indent=4,
        )

    print()
    print("=" * 70)
    print("ROBUSTNESS ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print("Generated files:")

    print(
        " - robustness_error_analysis.csv"
    )

    print(
        " - error_analysis_by_perturbation.csv"
    )

    print(
        " - error_analysis_by_polarity.csv"
    )

    print(
        " - error_analysis_by_aspect.csv"
    )

    print(
        " - prediction_flip_patterns.csv"
    )

    print(
        " - error_pattern_categories.csv"
    )

    print(
        " - high_confidence_failures.csv"
    )

    print(
        " - representative_failure_cases.csv"
    )

    print(
        " - robustness_error_analysis_summary.json"
    )

    print()
    print("Next step:")
    print(
        "Run targeted adversarial evaluation "
        "before finalizing Experiment 4, "
        "so the error analysis can also include "
        "Experiment 3 adversarial failures."
    )


if __name__ == "__main__":
    main()