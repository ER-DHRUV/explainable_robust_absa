from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# ROBUSTNESS EXPERIMENT 2
# BASELINE PERTURBATION EVALUATION
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_DIR = (
    PROJECT_ROOT
    / "results"
    / "transformer"
    / "distilbert_baseline"
)

MODEL_DIR = BASELINE_DIR / "best_model"

PERTURBATION_DIR = (
    BASELINE_DIR
    / "robustness"
    / "controlled_perturbations"
)

INPUT_FILE = (
    PERTURBATION_DIR
    / "perturbed_instances.csv"
)

OUTPUT_DIR = (
    BASELINE_DIR
    / "robustness"
    / "baseline_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LABEL HELPERS
# ============================================================

def normalize_polarity(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value not in LABEL_NAMES.values():
        return value

    return value


def polarity_to_label(value):
    value = normalize_polarity(value)

    mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    return mapping.get(value)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            "Best DistilBERT model not found:\n"
            f"{MODEL_DIR}\n\n"
            "Run the DistilBERT baseline training "
            "experiment first."
        )

    print()
    print("=" * 70)
    print("LOADING DISTILBERT BASELINE")
    print("=" * 70)

    print(
        f"Model directory:\n{MODEL_DIR}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            str(MODEL_DIR)
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    return (
        tokenizer,
        model,
        device,
    )


# ============================================================
# LOAD PERTURBATIONS
# ============================================================

def load_perturbations():

    print()
    print("=" * 70)
    print("LOADING CONTROLLED PERTURBATIONS")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Perturbation file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run controlled_perturbation_generation.py "
            "first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows: {len(df):,}"
    )

    required_columns = [
        "perturbation_id",
        "sentence_id",
        "aspect",
        "true_polarity",
        "perturbation_type",
        "original_sentence",
        "perturbed_sentence",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing
            )
        )

    df["true_polarity"] = (
        df["true_polarity"]
        .apply(normalize_polarity)
    )

    df["true_label"] = (
        df["true_polarity"]
        .apply(polarity_to_label)
    )

    invalid_labels = df["true_label"].isna().sum()

    if invalid_labels > 0:
        raise ValueError(
            f"Found {invalid_labels} rows with invalid "
            "true_polarity labels."
        )

    return df


# ============================================================
# MODEL PREDICTION
# ============================================================

@torch.no_grad()
def predict_batch(
    tokenizer,
    model,
    device,
    aspects,
    sentences,
):

    encoded = tokenizer(
        list(aspects),
        list(sentences),
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    outputs = model(
        **encoded
    )

    logits = outputs.logits

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    predictions = torch.argmax(
        probabilities,
        dim=-1,
    )

    confidence = torch.max(
        probabilities,
        dim=-1,
    ).values

    return (
        logits.cpu().numpy(),
        probabilities.cpu().numpy(),
        predictions.cpu().numpy(),
        confidence.cpu().numpy(),
    )


# ============================================================
# RUN MODEL ON DATA
# ============================================================

def run_predictions(
    df,
    tokenizer,
    model,
    device,
    sentence_column,
    prefix,
):

    all_logits = []
    all_probabilities = []
    all_predictions = []
    all_confidences = []

    total = len(df)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = df.iloc[
            start:end
        ]

        (
            logits,
            probabilities,
            predictions,
            confidence,
        ) = predict_batch(
            tokenizer,
            model,
            device,
            batch["aspect"].astype(str),
            batch[
                sentence_column
            ].astype(str),
        )

        all_logits.append(
            logits
        )

        all_probabilities.append(
            probabilities
        )

        all_predictions.append(
            predictions
        )

        all_confidences.append(
            confidence
        )

        if (
            end == total
            or end % 500 == 0
        ):
            print(
                f"{prefix}: "
                f"{end:,}/{total:,}"
            )

    return (
        np.concatenate(
            all_logits,
            axis=0,
        ),
        np.concatenate(
            all_probabilities,
            axis=0,
        ),
        np.concatenate(
            all_predictions,
            axis=0,
        ),
        np.concatenate(
            all_confidences,
            axis=0,
        ),
    )


# ============================================================
# BUILD ORIGINAL INSTANCE TABLE
# ============================================================

def build_original_dataframe(df):

    columns = [
        "sentence_id",
        "aspect",
        "true_polarity",
        "true_label",
        "original_sentence",
    ]

    original_df = (
        df[columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return original_df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions,
):

    return {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything()

    print("=" * 70)
    print("ROBUSTNESS EXPERIMENT 2")
    print("BASELINE PERTURBATION EVALUATION")
    print("=" * 70)

    print(
        f"Input:\n{INPUT_FILE}"
    )

    print(
        f"Model:\n{MODEL_DIR}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_perturbations()

    (
        tokenizer,
        model,
        device,
    ) = load_model()

    # --------------------------------------------------------
    # Original instances
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING ORIGINAL INSTANCES")
    print("=" * 70)

    original_df = (
        build_original_dataframe(df)
    )

    print(
        f"Unique original instances: "
        f"{len(original_df):,}"
    )

    # --------------------------------------------------------
    # Predict original sentences
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREDICTING ORIGINAL SENTENCES")
    print("=" * 70)

    (
        original_logits,
        original_probabilities,
        original_predictions,
        original_confidences,
    ) = run_predictions(
        original_df,
        tokenizer,
        model,
        device,
        "original_sentence",
        "Original",
    )

    original_df[
        "original_predicted_label"
    ] = original_predictions

    original_df[
        "original_predicted_polarity"
    ] = [
        LABEL_NAMES[int(label)]
        for label in original_predictions
    ]

    original_df[
        "original_confidence"
    ] = original_confidences

    original_df[
        "original_correct"
    ] = (
        original_df["true_label"]
        ==
        original_df["original_predicted_label"]
    )

    # --------------------------------------------------------
    # Original baseline metrics
    # --------------------------------------------------------

    original_metrics = calculate_metrics(
        original_df["true_label"].values,
        original_df[
            "original_predicted_label"
        ].values,
    )

    print()
    print(
        "Original performance:"
    )

    print(
        f"Accuracy   : "
        f"{original_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro-F1   : "
        f"{original_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted-F1: "
        f"{original_metrics['weighted_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Predict perturbed sentences
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREDICTING PERTURBED SENTENCES")
    print("=" * 70)

    (
        perturbed_logits,
        perturbed_probabilities,
        perturbed_predictions,
        perturbed_confidences,
    ) = run_predictions(
        df,
        tokenizer,
        model,
        device,
        "perturbed_sentence",
        "Perturbed",
    )

    df[
        "perturbed_predicted_label"
    ] = perturbed_predictions

    df[
        "perturbed_predicted_polarity"
    ] = [
        LABEL_NAMES[int(label)]
        for label in perturbed_predictions
    ]

    df[
        "perturbed_confidence"
    ] = perturbed_confidences

    df[
        "perturbed_correct"
    ] = (
        df["true_label"]
        ==
        df["perturbed_predicted_label"]
    )

    # --------------------------------------------------------
    # Attach original prediction
    # --------------------------------------------------------

    original_lookup = (
        original_df[
            [
                "sentence_id",
                "aspect",
                "original_predicted_label",
                "original_predicted_polarity",
                "original_confidence",
                "original_correct",
            ]
        ]
    )

    df = df.merge(
        original_lookup,
        on=[
            "sentence_id",
            "aspect",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Prediction consistency
    # --------------------------------------------------------

    df[
        "prediction_consistent"
    ] = (
        df["original_predicted_label"]
        ==
        df["perturbed_predicted_label"]
    )

    df[
        "prediction_flip"
    ] = (
        ~df["prediction_consistent"]
    )

    # --------------------------------------------------------
    # Correctness changes
    # --------------------------------------------------------

    df[
        "correctness_preserved"
    ] = (
        df["original_correct"]
        ==
        df["perturbed_correct"]
    )

    df[
        "correct_to_incorrect"
    ] = (
        df["original_correct"]
        &
        ~df["perturbed_correct"]
    )

    df[
        "incorrect_to_correct"
    ] = (
        ~df["original_correct"]
        &
        df["perturbed_correct"]
    )

    # --------------------------------------------------------
    # Confidence change
    # --------------------------------------------------------

    df[
        "confidence_change"
    ] = (
        df["perturbed_confidence"]
        -
        df["original_confidence"]
    )

    df[
        "absolute_confidence_change"
    ] = (
        df["confidence_change"]
        .abs()
    )

    # --------------------------------------------------------
    # Probability changes
    # --------------------------------------------------------

    # Build a reliable lookup indexed by
    # sentence_id + aspect.

    original_probability_lookup = {}

    for index, row in original_df.iterrows():

        key = (
            row["sentence_id"],
            row["aspect"],
        )

        original_probability_lookup[key] = (
            original_probabilities[index]
        )

    for label_id, polarity in LABEL_NAMES.items():

        original_column = (
            f"original_probability_{polarity}"
        )

        perturbed_column = (
            f"perturbed_probability_{polarity}"
        )

        df[original_column] = [
            original_probability_lookup[
                (
                    row["sentence_id"],
                    row["aspect"],
                )
            ][label_id]
            for _, row in df.iterrows()
        ]

        df[perturbed_column] = (
            perturbed_probabilities[:, label_id]
        )

    # --------------------------------------------------------
    # Probability change for original predicted class
    # --------------------------------------------------------

    df[
        "predicted_class_probability_change"
    ] = np.nan

    for index in range(len(df)):

        original_prediction = int(
            df.iloc[index][
                "original_predicted_label"
            ]
        )

        polarity = LABEL_NAMES[
            original_prediction
        ]

        original_probability = df.iloc[index][
            f"original_probability_{polarity}"
        ]

        perturbed_probability = df.iloc[index][
            f"perturbed_probability_{polarity}"
        ]

        df.loc[
            df.index[index],
            "predicted_class_probability_change",
        ] = (
            perturbed_probability
            -
            original_probability
        )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    detailed_columns = [
        "perturbation_id",
        "sentence_id",
        "aspect",
        "true_polarity",
        "perturbation_type",
        "original_sentence",
        "perturbed_sentence",

        "original_predicted_label",
        "original_predicted_polarity",
        "original_confidence",
        "original_correct",

        "perturbed_predicted_label",
        "perturbed_predicted_polarity",
        "perturbed_confidence",
        "perturbed_correct",

        "prediction_consistent",
        "prediction_flip",

        "correctness_preserved",
        "correct_to_incorrect",
        "incorrect_to_correct",

        "confidence_change",
        "absolute_confidence_change",

        "predicted_class_probability_change",

        "original_probability_negative",
        "original_probability_neutral",
        "original_probability_positive",

        "perturbed_probability_negative",
        "perturbed_probability_neutral",
        "perturbed_probability_positive",
    ]

    detailed_output = (
        OUTPUT_DIR
        / "baseline_perturbation_results.csv"
    )

    df[
        detailed_columns
    ].to_csv(
        detailed_output,
        index=False,
    )

    print()
    print(
        f"Saved:\n{detailed_output}"
    )

    # ========================================================
    # OVERALL ROBUSTNESS
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL ROBUSTNESS")
    print("=" * 70)

    consistency_rate = (
        df["prediction_consistent"]
        .mean()
    )

    flip_rate = (
        df["prediction_flip"]
        .mean()
    )

    correct_to_incorrect_rate = (
        df["correct_to_incorrect"]
        .mean()
    )

    incorrect_to_correct_rate = (
        df["incorrect_to_correct"]
        .mean()
    )

    mean_confidence_change = (
        df["confidence_change"]
        .mean()
    )

    mean_absolute_confidence_change = (
        df["absolute_confidence_change"]
        .mean()
    )

    print(
        f"Perturbed instances        : "
        f"{len(df):,}"
    )

    print(
        f"Prediction consistency     : "
        f"{consistency_rate:.4f}"
    )

    print(
        f"Prediction flip rate       : "
        f"{flip_rate:.4f}"
    )

    print(
        f"Correct → incorrect rate   : "
        f"{correct_to_incorrect_rate:.4f}"
    )

    print(
        f"Incorrect → correct rate   : "
        f"{incorrect_to_correct_rate:.4f}"
    )

    print(
        f"Mean confidence change     : "
        f"{mean_confidence_change:.4f}"
    )

    print(
        f"Mean absolute confidence Δ : "
        f"{mean_absolute_confidence_change:.4f}"
    )

    # ========================================================
    # PERTURBED METRICS
    # ========================================================

    perturbed_metrics = calculate_metrics(
        df["true_label"].values,
        df[
            "perturbed_predicted_label"
        ].values,
    )

    print()
    print("Perturbed performance:")

    print(
        f"Accuracy   : "
        f"{perturbed_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro-F1   : "
        f"{perturbed_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted-F1: "
        f"{perturbed_metrics['weighted_f1']:.4f}"
    )

    # ========================================================
    # DEGRADATION
    # ========================================================

    macro_f1_degradation = (
        original_metrics["macro_f1"]
        -
        perturbed_metrics["macro_f1"]
    )

    accuracy_degradation = (
        original_metrics["accuracy"]
        -
        perturbed_metrics["accuracy"]
    )

    print()
    print(
        f"Macro-F1 degradation : "
        f"{macro_f1_degradation:.4f}"
    )

    print(
        f"Accuracy degradation : "
        f"{accuracy_degradation:.4f}"
    )

    # ========================================================
    # PERTURBATION-TYPE ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("ROBUSTNESS BY PERTURBATION TYPE")
    print("=" * 70)

    perturbation_rows = []

    for perturbation_type, group in df.groupby(
        "perturbation_type",
        sort=True,
    ):

        labels = group[
            "true_label"
        ].values

        predictions = group[
            "perturbed_predicted_label"
        ].values

        metrics = calculate_metrics(
            labels,
            predictions,
        )

        perturbation_rows.append(
            {
                "perturbation_type": (
                    perturbation_type
                ),

                "instances": int(
                    len(group)
                ),

                "accuracy": metrics[
                    "accuracy"
                ],

                "macro_f1": metrics[
                    "macro_f1"
                ],

                "weighted_f1": metrics[
                    "weighted_f1"
                ],

                "prediction_consistency": (
                    group[
                        "prediction_consistent"
                    ].mean()
                ),

                "prediction_flip_rate": (
                    group[
                        "prediction_flip"
                    ].mean()
                ),

                "correct_to_incorrect_rate": (
                    group[
                        "correct_to_incorrect"
                    ].mean()
                ),

                "incorrect_to_correct_rate": (
                    group[
                        "incorrect_to_correct"
                    ].mean()
                ),

                "mean_original_confidence": (
                    group[
                        "original_confidence"
                    ].mean()
                ),

                "mean_perturbed_confidence": (
                    group[
                        "perturbed_confidence"
                    ].mean()
                ),

                "mean_confidence_change": (
                    group[
                        "confidence_change"
                    ].mean()
                ),

                "mean_absolute_confidence_change": (
                    group[
                        "absolute_confidence_change"
                    ].mean()
                ),
            }
        )

    perturbation_df = pd.DataFrame(
        perturbation_rows
    )

    perturbation_output = (
        OUTPUT_DIR
        / "robustness_by_perturbation_type.csv"
    )

    perturbation_df.to_csv(
        perturbation_output,
        index=False,
    )

    print(
        perturbation_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved:\n{perturbation_output}"
    )

    # ========================================================
    # POLARITY ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("ROBUSTNESS BY TRUE POLARITY")
    print("=" * 70)

    polarity_rows = []

    for polarity, group in df.groupby(
        "true_polarity",
        sort=True,
    ):

        metrics = calculate_metrics(
            group["true_label"].values,
            group[
                "perturbed_predicted_label"
            ].values,
        )

        polarity_rows.append(
            {
                "true_polarity": polarity,

                "instances": int(
                    len(group)
                ),

                "accuracy": metrics[
                    "accuracy"
                ],

                "macro_f1": metrics[
                    "macro_f1"
                ],

                "prediction_consistency": (
                    group[
                        "prediction_consistent"
                    ].mean()
                ),

                "prediction_flip_rate": (
                    group[
                        "prediction_flip"
                    ].mean()
                ),

                "correct_to_incorrect_rate": (
                    group[
                        "correct_to_incorrect"
                    ].mean()
                ),

                "incorrect_to_correct_rate": (
                    group[
                        "incorrect_to_correct"
                    ].mean()
                ),

                "mean_confidence_change": (
                    group[
                        "confidence_change"
                    ].mean()
                ),

                "mean_absolute_confidence_change": (
                    group[
                        "absolute_confidence_change"
                    ].mean()
                ),
            }
        )

    polarity_df = pd.DataFrame(
        polarity_rows
    )

    polarity_output = (
        OUTPUT_DIR
        / "robustness_by_polarity.csv"
    )

    polarity_df.to_csv(
        polarity_output,
        index=False,
    )

    print(
        polarity_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved:\n{polarity_output}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 70)
    print("PERTURBED CONFUSION MATRIX")
    print("=" * 70)

    perturbed_labels = df[
        "true_label"
    ].values

    perturbed_predictions = df[
        "perturbed_predicted_label"
    ].values

    perturbed_cm = confusion_matrix(
        perturbed_labels,
        perturbed_predictions,
        labels=[0, 1, 2],
    )

    print(
        perturbed_cm
    )

    confusion_output = (
        OUTPUT_DIR
        / "perturbed_confusion_matrix.csv"
    )

    confusion_df = pd.DataFrame(
        perturbed_cm,
        index=[
            "true_negative",
            "true_neutral",
            "true_positive",
        ],
        columns=[
            "pred_negative",
            "pred_neutral",
            "pred_positive",
        ],
    )

    confusion_df.to_csv(
        confusion_output
    )

    print(
        f"\nSaved:\n{confusion_output}"
    )

    # ========================================================
    # FLIP ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("PREDICTION FLIP ANALYSIS")
    print("=" * 70)

    flipped = df[
        df["prediction_flip"]
    ].copy()

    print(
        f"Total flips: {len(flipped):,}"
    )

    if len(flipped) > 0:

        flip_rows = []

        for (
            original_polarity,
            perturbed_group,
        ) in flipped.groupby(
            "original_predicted_polarity"
        ):

            for (
                perturbed_polarity,
                group,
            ) in perturbed_group.groupby(
                "perturbed_predicted_polarity"
            ):

                flip_rows.append(
                    {
                        "original_prediction": (
                            original_polarity
                        ),

                        "perturbed_prediction": (
                            perturbed_polarity
                        ),

                        "instances": int(
                            len(group)
                        ),

                        "mean_confidence_change": (
                            group[
                                "confidence_change"
                            ].mean()
                        ),
                    }
                )

        flip_df = pd.DataFrame(
            flip_rows
        )

    else:

        flip_df = pd.DataFrame(
            columns=[
                "original_prediction",
                "perturbed_prediction",
                "instances",
                "mean_confidence_change",
            ]
        )

    flip_output = (
        OUTPUT_DIR
        / "prediction_flip_analysis.csv"
    )

    flip_df.to_csv(
        flip_output,
        index=False,
    )

    if len(flip_df) > 0:

        print(
            flip_df.to_string(
                index=False
            )
        )

    print(
        f"\nSaved:\n{flip_output}"
    )

    # ========================================================
    # HIGH-CONFIDENCE FLIPS
    # ========================================================

    print()
    print("=" * 70)
    print("HIGH-CONFIDENCE PREDICTION FLIPS")
    print("=" * 70)

    high_confidence_flips = (
        df[
            df["prediction_flip"]
        ]
        .sort_values(
            by="original_confidence",
            ascending=False,
        )
        .head(50)
    )

    high_confidence_columns = [
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
        "original_sentence",
        "perturbed_sentence",
    ]

    high_confidence_output = (
        OUTPUT_DIR
        / "high_confidence_prediction_flips.csv"
    )

    high_confidence_flips[
        high_confidence_columns
    ].to_csv(
        high_confidence_output,
        index=False,
    )

    if len(high_confidence_flips) > 0:
        print(
            high_confidence_flips[
                high_confidence_columns
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No prediction flips found."
        )

    print(
        f"\nSaved:\n{high_confidence_output}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary = {

        "experiment": (
            "Robustness Experiment 2"
        ),

        "name": (
            "Baseline Perturbation Evaluation"
        ),

        "model": (
            "DistilBERT baseline best_model"
        ),

        "model_directory": str(
            MODEL_DIR
        ),

        "input_file": str(
            INPUT_FILE
        ),

        "device": str(
            device
        ),

        "original_instances": int(
            len(original_df)
        ),

        "perturbed_instances": int(
            len(df)
        ),

        "original_metrics": (
            original_metrics
        ),

        "perturbed_metrics": (
            perturbed_metrics
        ),

        "prediction_consistency": float(
            consistency_rate
        ),

        "prediction_flip_rate": float(
            flip_rate
        ),

        "correct_to_incorrect_rate": float(
            correct_to_incorrect_rate
        ),

        "incorrect_to_correct_rate": float(
            incorrect_to_correct_rate
        ),

        "mean_confidence_change": float(
            mean_confidence_change
        ),

        "mean_absolute_confidence_change": float(
            mean_absolute_confidence_change
        ),

        "macro_f1_degradation": float(
            macro_f1_degradation
        ),

        "accuracy_degradation": float(
            accuracy_degradation
        ),
    }

    summary_output = (
        OUTPUT_DIR
        / "robustness_summary.json"
    )

    with open(
        summary_output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print("=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    print(
        json.dumps(
            summary,
            indent=4,
        )
    )

    print(
        f"\nSaved:\n{summary_output}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "ROBUSTNESS EXPERIMENT 2 COMPLETE"
    )
    print("=" * 70)

    print()
    print("Generated files:")

    generated_files = [
        "baseline_perturbation_results.csv",
        "robustness_by_perturbation_type.csv",
        "robustness_by_polarity.csv",
        "perturbed_confusion_matrix.csv",
        "prediction_flip_analysis.csv",
        "high_confidence_prediction_flips.csv",
        "robustness_summary.json",
    ]

    for filename in generated_files:
        print(
            f" - {filename}"
        )


if __name__ == "__main__":
    main()
