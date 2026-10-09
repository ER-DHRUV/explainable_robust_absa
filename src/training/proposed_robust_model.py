from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# EXPERIMENT 5
# PROPOSED ROBUST DISTILBERT MODEL
#
# Strategy:
#   1. Load original training data.
#   2. Generate label-preserving perturbations on TRAIN only.
#   3. Combine clean + perturbed training instances.
#   4. Fine-tune DistilBERT.
#   5. Evaluate on clean validation/test data.
#   6. Evaluate on the EXISTING 7,386-instance robustness set.
#
# IMPORTANT:
# The existing perturbed_instances.csv is NOT used for training.
# It is reserved for robustness evaluation.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

NUM_EPOCHS = 5

WARMUP_RATIO = 0.10

AUGMENTATION_RATIO = 1.0

# Number of augmented copies generated per selected
# training instance.
#
# 1.0 means approximately one perturbed example per
# original training example.
AUGMENTATIONS_PER_INSTANCE = 1

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

BASELINE_DIR = (
    PROJECT_ROOT
    / "results"
    / "transformer"
    / "distilbert_baseline"
)

ROBUSTNESS_EVAL_FILE = (
    BASELINE_DIR
    / "robustness"
    / "controlled_perturbations"
    / "perturbed_instances.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "transformer"
    / "proposed_robust_model"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


# ============================================================
# DATA LOADING
# ============================================================

def load_split(split: str) -> pd.DataFrame:

    path = DATA_DIR / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "sentence_id",
        "sentence",
        "aspect",
        "polarity",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{split}: missing columns "
            f"{sorted(missing)}"
        )

    df["label"] = df["label"].astype(int)

    return df


# ============================================================
# LABEL-PRESERVING TRAINING PERTURBATIONS
#
# These correspond to the perturbation families visible in
# your existing controlled perturbation dataset:
#
#   - context_insertion
#   - neutral_suffix
#   - irrelevant_noise
#   - punctuation_variation
#   - whitespace_variation
#
# They are deliberately conservative because the objective is
# to create label-preserving training examples.
# ============================================================

def perturb_context_insertion(sentence: str) -> str:

    prefixes = [
        "From the review, ",
        "In this review, ",
        "According to the customer, ",
        "In the customer's account, ",
    ]

    prefix = random.choice(prefixes)

    return prefix + sentence


def perturb_neutral_suffix(sentence: str) -> str:

    suffixes = [
        " The visit was on a weekday.",
        " The review describes a restaurant visit.",
        " This was part of the customer's overall experience.",
        " The customer included this detail in the review.",
    ]

    suffix = random.choice(suffixes)

    return sentence.rstrip() + suffix


def perturb_irrelevant_noise(sentence: str) -> str:

    variants = [
        " indeed",
        " overall",
        " as noted",
        " in particular",
    ]

    noise = random.choice(variants)

    stripped = sentence.rstrip()

    if stripped.endswith("."):
        return stripped[:-1] + noise + "."

    if stripped.endswith("!"):
        return stripped[:-1] + noise + "!"

    if stripped.endswith("?"):
        return stripped[:-1] + noise + "?"

    return stripped + noise


def perturb_punctuation(sentence: str) -> str:

    stripped = sentence.rstrip()

    if stripped.endswith("."):
        return stripped[:-1] + "!"

    if stripped.endswith("!"):
        return stripped[:-1] + "."

    if stripped.endswith("?"):
        return stripped[:-1] + "."

    return stripped + "."


def perturb_whitespace(sentence: str) -> str:

    words = sentence.split()

    if len(words) < 3:
        return sentence

    # Introduce harmless extra whitespace at a deterministic
    # random location.
    index = random.randint(
        1,
        len(words) - 1,
    )

    return (
        " ".join(words[:index])
        + "  "
        + " ".join(words[index:])
    )


PERTURBATION_FUNCTIONS = {
    "context_insertion": perturb_context_insertion,
    "neutral_suffix": perturb_neutral_suffix,
    "irrelevant_noise": perturb_irrelevant_noise,
    "punctuation_variation": perturb_punctuation,
    "whitespace_variation": perturb_whitespace,
}


# ============================================================
# GENERATE ROBUST TRAINING DATA
# ============================================================

def generate_training_perturbations(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("GENERATING TRAINING PERTURBATIONS")
    print("=" * 70)

    print(
        f"Original training instances: "
        f"{len(df):,}"
    )

    target_count = int(
        len(df)
        * AUGMENTATION_RATIO
        * AUGMENTATIONS_PER_INSTANCE
    )

    print(
        f"Target augmented instances: "
        f"{target_count:,}"
    )

    rows = []

    perturbation_types = list(
        PERTURBATION_FUNCTIONS.keys()
    )

    for i in range(target_count):

        source_index = i % len(df)

        row = df.iloc[source_index]

        perturbation_type = (
            perturbation_types[
                i % len(perturbation_types)
            ]
        )

        perturbation_function = (
            PERTURBATION_FUNCTIONS[
                perturbation_type
            ]
        )

        original_sentence = str(
            row["sentence"]
        )

        perturbed_sentence = (
            perturbation_function(
                original_sentence
            )
        )

        rows.append(
            {
                "sentence_id": (
                    f"ROBUST_TRAIN_{i:07d}"
                ),

                "source_sentence_id": (
                    row["sentence_id"]
                ),

                "sentence": perturbed_sentence,

                "aspect": str(
                    row["aspect"]
                ),

                "polarity": str(
                    row["polarity"]
                ),

                "label": int(
                    row["label"]
                ),

                "perturbation_type": (
                    perturbation_type
                ),

                "original_sentence": (
                    original_sentence
                ),
            }
        )

    augmented_df = pd.DataFrame(rows)

    print()
    print(
        "Generated perturbations:"
    )

    print(
        augmented_df[
            "perturbation_type"
        ].value_counts()
        .sort_index()
        .to_string()
    )

    return augmented_df


# ============================================================
# BUILD ROBUST TRAINING DATASET
# ============================================================

def build_training_dataframe(
    train_df: pd.DataFrame,
) -> pd.DataFrame:

    augmented_df = (
        generate_training_perturbations(
            train_df
        )
    )

    clean_df = train_df[
        [
            "sentence_id",
            "sentence",
            "aspect",
            "polarity",
            "label",
        ]
    ].copy()

    clean_df["training_source"] = "clean"

    augmented_training_df = augmented_df[
        [
            "sentence_id",
            "sentence",
            "aspect",
            "polarity",
            "label",
        ]
    ].copy()

    augmented_training_df[
        "training_source"
    ] = "perturbed"

    combined = pd.concat(
        [
            clean_df,
            augmented_training_df,
        ],
        ignore_index=True,
    )

    # Shuffle while preserving reproducibility.
    combined = combined.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    print()
    print("=" * 70)
    print("ROBUST TRAINING DATASET")
    print("=" * 70)

    print(
        f"Clean instances     : "
        f"{len(clean_df):,}"
    )

    print(
        f"Perturbed instances : "
        f"{len(augmented_training_df):,}"
    )

    print(
        f"Total instances     : "
        f"{len(combined):,}"
    )

    print()
    print(
        combined[
            "training_source"
        ].value_counts()
        .to_string()
    )

    return combined


# ============================================================
# HUGGING FACE DATASET
# ============================================================

def dataframe_to_dataset(
    df: pd.DataFrame,
) -> Dataset:

    data = {
        "sentence": (
            df["sentence"]
            .astype(str)
            .tolist()
        ),

        "aspect": (
            df["aspect"]
            .astype(str)
            .tolist()
        ),

        "labels": (
            df["label"]
            .astype(int)
            .tolist()
        ),
    }

    return Dataset.from_dict(data)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
) -> Dataset:

    def tokenize_batch(batch):

        return tokenizer(
            batch["aspect"],
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[
            "sentence",
            "aspect",
        ],
        desc="Tokenizing",
    )

    return tokenized


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    eval_prediction,
):

    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),

        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_dataset(
    trainer: Trainer,
    dataset: Dataset,
    dataframe: pd.DataFrame,
    split_name: str,
    output_prefix: str | None = None,
):

    prediction_output = trainer.predict(
        dataset
    )

    logits = prediction_output.predictions

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    labels = np.asarray(
        dataframe["label"].values
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 70)
    print(
        f"PROPOSED ROBUST MODEL — "
        f"{split_name.upper()}"
    )
    print("=" * 70)

    print(
        f"Instances  : {len(labels):,}"
    )

    print(
        f"Accuracy   : {accuracy:.4f}"
    )

    print(
        f"Macro-F1   : {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1: {weighted_f1:.4f}"
    )

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            LABEL_NAMES[0],
            LABEL_NAMES[1],
            LABEL_NAMES[2],
        ],
        digits=4,
        zero_division=0,
    )

    print()
    print(report)

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1, 2],
    )

    print("Confusion matrix:")
    print(cm)

    metrics = {
        "accuracy": float(accuracy),

        "macro_f1": float(macro_f1),

        "weighted_f1": float(
            weighted_f1
        ),

        "classification_report": (
            classification_report(
                labels,
                predictions,
                labels=[0, 1, 2],
                target_names=[
                    LABEL_NAMES[0],
                    LABEL_NAMES[1],
                    LABEL_NAMES[2],
                ],
                output_dict=True,
                zero_division=0,
            )
        ),

        "confusion_matrix": (
            cm.tolist()
        ),
    }

    # --------------------------------------------------------
    # Save predictions when requested
    # --------------------------------------------------------

    if output_prefix is not None:

        output = dataframe.copy()

        output["true_label"] = labels

        output[
            "predicted_label"
        ] = predictions

        output[
            "true_polarity"
        ] = (
            output["true_label"]
            .map(LABEL_NAMES)
        )

        output[
            "predicted_polarity"
        ] = (
            output["predicted_label"]
            .map(LABEL_NAMES)
        )

        output["correct"] = (
            output["true_label"]
            ==
            output["predicted_label"]
        )

        prediction_path = (
            OUTPUT_DIR
            / f"{output_prefix}_predictions.csv"
        )

        output.to_csv(
            prediction_path,
            index=False,
        )

        print(
            f"\nSaved predictions:\n"
            f"{prediction_path}"
        )

    return metrics, predictions


# ============================================================
# ROBUSTNESS EVALUATION
# ============================================================

def load_robustness_dataset():

    if not ROBUSTNESS_EVAL_FILE.exists():

        raise FileNotFoundError(
            "Robustness evaluation file not found:\n"
            f"{ROBUSTNESS_EVAL_FILE}"
        )

    df = pd.read_csv(
        ROBUSTNESS_EVAL_FILE
    )

    required_columns = [
        "perturbation_id",
        "sentence_id",
        "aspect",
        "true_polarity",
        "original_sentence",
        "perturbed_sentence",
        "perturbation_type",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Robustness dataset is missing:\n"
            + "\n".join(
                f" - {column}"
                for column in missing
            )
        )

    polarity_to_label = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    df["true_polarity"] = (
        df["true_polarity"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["label"] = (
        df["true_polarity"]
        .map(polarity_to_label)
    )

    if df["label"].isna().any():

        raise ValueError(
            "Invalid polarity found in robustness dataset."
        )

    df["label"] = df["label"].astype(int)

    return df


# ============================================================
# ROBUSTNESS EVALUATION
# ============================================================

def evaluate_robustness(
    trainer: Trainer,
    tokenizer,
    robustness_df: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("ROBUSTNESS EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Build perturbed evaluation dataset
    # --------------------------------------------------------

    perturbed_df = robustness_df[
        [
            "perturbation_id",
            "sentence_id",
            "aspect",
            "true_polarity",
            "perturbed_sentence",
            "label",
            "perturbation_type",
        ]
    ].copy()

    perturbed_df = perturbed_df.rename(
        columns={
            "perturbed_sentence": "sentence"
        }
    )

    perturbed_hf = (
        dataframe_to_dataset(
            perturbed_df
        )
    )

    perturbed_hf = tokenize_dataset(
        perturbed_hf,
        tokenizer,
    )

    (
        perturbed_metrics,
        perturbed_predictions,
    ) = evaluate_dataset(
        trainer,
        perturbed_hf,
        perturbed_df,
        "robustness_perturbed",
        "robustness",
    )

    perturbed_df[
        "predicted_label"
    ] = perturbed_predictions

    perturbed_df[
        "predicted_polarity"
    ] = (
        perturbed_df[
            "predicted_label"
        ].map(LABEL_NAMES)
    )

    perturbed_df[
        "correct"
    ] = (
        perturbed_df["label"]
        ==
        perturbed_df[
            "predicted_label"
        ]
    )

    # --------------------------------------------------------
    # Analyze consistency against original sentence
    # --------------------------------------------------------

    original_df = robustness_df[
        [
            "sentence_id",
            "aspect",
            "true_polarity",
            "original_sentence",
            "label",
        ]
    ].drop_duplicates(
        subset=[
            "sentence_id",
            "aspect",
        ]
    ).reset_index(
        drop=True
    )

    original_hf = dataframe_to_dataset(
        original_df.rename(
            columns={
                "original_sentence": "sentence"
            }
        )
    )

    original_hf = tokenize_dataset(
        original_hf,
        tokenizer,
    )

    (
        original_metrics,
        original_predictions,
    ) = evaluate_dataset(
        trainer,
        original_hf,
        original_df.rename(
            columns={
                "original_sentence": "sentence"
            }
        ),
        "robustness_original",
        "robustness_original",
    )

    original_df[
        "original_predicted_label"
    ] = original_predictions

    original_df[
        "original_predicted_polarity"
    ] = (
        original_df[
            "original_predicted_label"
        ].map(LABEL_NAMES)
    )

    # --------------------------------------------------------
    # Attach original prediction
    # --------------------------------------------------------

    lookup = original_df[
        [
            "sentence_id",
            "aspect",
            "original_predicted_label",
            "original_predicted_polarity",
        ]
    ]

    result = perturbed_df.merge(
        lookup,
        on=[
            "sentence_id",
            "aspect",
        ],
        how="left",
        validate="many_to_one",
    )

    result[
        "prediction_consistent"
    ] = (
        result[
            "original_predicted_label"
        ]
        ==
        result[
            "predicted_label"
        ]
    )

    result[
        "prediction_flip"
    ] = ~result[
        "prediction_consistent"
    ]

    result[
        "correct_to_incorrect"
    ] = (
        (
            result[
                "original_predicted_label"
            ]
            ==
            result["label"]
        )
        &
        (
            result[
                "predicted_label"
            ]
            !=
            result["label"]
        )
    )

    result[
        "incorrect_to_correct"
    ] = (
        (
            result[
                "original_predicted_label"
            ]
            !=
            result["label"]
        )
        &
        (
            result[
                "predicted_label"
            ]
            ==
            result["label"]
        )
    )

    # --------------------------------------------------------
    # Overall robustness statistics
    # --------------------------------------------------------

    consistency = (
        result[
            "prediction_consistent"
        ].mean()
    )

    flip_rate = (
        result[
            "prediction_flip"
        ].mean()
    )

    correct_to_incorrect = (
        result[
            "correct_to_incorrect"
        ].mean()
    )

    incorrect_to_correct = (
        result[
            "incorrect_to_correct"
        ].mean()
    )

    print()
    print(
        f"Robustness instances      : "
        f"{len(result):,}"
    )

    print(
        f"Prediction consistency    : "
        f"{consistency:.4f}"
    )

    print(
        f"Prediction flip rate      : "
        f"{flip_rate:.4f}"
    )

    print(
        f"Correct -> incorrect rate : "
        f"{correct_to_incorrect:.4f}"
    )

    print(
        f"Incorrect -> correct rate : "
        f"{incorrect_to_correct:.4f}"
    )

    # --------------------------------------------------------
    # By perturbation type
    # --------------------------------------------------------

    type_rows = []

    for perturbation_type, group in result.groupby(
        "perturbation_type",
        sort=True,
    ):

        type_rows.append(
            {
                "perturbation_type": (
                    perturbation_type
                ),

                "instances": int(
                    len(group)
                ),

                "accuracy": float(
                    group["correct"].mean()
                ),

                "prediction_consistency": float(
                    group[
                        "prediction_consistent"
                    ].mean()
                ),

                "prediction_flip_rate": float(
                    group[
                        "prediction_flip"
                    ].mean()
                ),

                "correct_to_incorrect_rate": float(
                    group[
                        "correct_to_incorrect"
                    ].mean()
                ),

                "incorrect_to_correct_rate": float(
                    group[
                        "incorrect_to_correct"
                    ].mean()
                ),
            }
        )

    type_df = pd.DataFrame(
        type_rows
    )

    type_output = (
        OUTPUT_DIR
        / "robustness_by_perturbation_type.csv"
    )

    type_df.to_csv(
        type_output,
        index=False,
    )

    print()
    print(
        type_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Detailed robustness results
    # --------------------------------------------------------

    result_output = (
        OUTPUT_DIR
        / "robustness_predictions.csv"
    )

    result.to_csv(
        result_output,
        index=False,
    )

    print(
        f"\nSaved:\n{result_output}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment": "Experiment 5",

        "name": (
            "Proposed Robust DistilBERT Model"
        ),

        "training_strategy": (
            "Clean training data plus "
            "label-preserving training perturbations"
        ),

        "model": MODEL_NAME,

        "seed": SEED,

        "training_augmentation_ratio": (
            AUGMENTATION_RATIO
        ),

        "augmentations_per_instance": (
            AUGMENTATIONS_PER_INSTANCE
        ),

        "robustness_instances": int(
            len(result)
        ),

        "original_metrics": (
            original_metrics
        ),

        "perturbed_metrics": (
            perturbed_metrics
        ),

        "prediction_consistency": float(
            consistency
        ),

        "prediction_flip_rate": float(
            flip_rate
        ),

        "correct_to_incorrect_rate": float(
            correct_to_incorrect
        ),

        "incorrect_to_correct_rate": float(
            incorrect_to_correct
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
    print(
        f"Saved:\n{summary_output}"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything()

    print("=" * 70)
    print("EXPERIMENT 5")
    print("PROPOSED ROBUST DISTILBERT MODEL")
    print("=" * 70)

    print(
        f"Model        : {MODEL_NAME}"
    )

    print(
        f"Max length   : {MAX_LENGTH}"
    )

    print(
        f"Train batch  : {TRAIN_BATCH_SIZE}"
    )

    print(
        f"Accumulation : {GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Effective BS : "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Epochs       : {NUM_EPOCHS}"
    )

    print(
        f"Seed         : {SEED}"
    )

    print(
        f"CUDA         : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM         : "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df = load_split("train")

    val_df = load_split("val")

    test_df = load_split("test")

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        f"Train: {len(train_df):,}"
    )

    print(
        f"Val  : {len(val_df):,}"
    )

    print(
        f"Test : {len(test_df):,}"
    )

    # ========================================================
    # BUILD ROBUST TRAINING DATA
    # ========================================================

    robust_train_df = (
        build_training_dataframe(
            train_df
        )
    )

    # Save the exact training augmentation manifest.
    training_manifest = (
        OUTPUT_DIR
        / "robust_training_data.csv"
    )

    robust_train_df.to_csv(
        training_manifest,
        index=False,
    )

    print()
    print(
        f"Training manifest saved:\n"
        f"{training_manifest}"
    )

    # ========================================================
    # HUGGING FACE DATASETS
    # ========================================================

    train_dataset = (
        dataframe_to_dataset(
            robust_train_df
        )
    )

    val_dataset = (
        dataframe_to_dataset(
            val_df
        )
    )

    test_dataset = (
        dataframe_to_dataset(
            test_df
        )
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print()
    print("Loading tokenizer...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    # ========================================================
    # TOKENIZATION
    # ========================================================

    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
    )

    val_dataset = tokenize_dataset(
        val_dataset,
        tokenizer,
    )

    test_dataset = tokenize_dataset(
        test_dataset,
        tokenizer,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("Loading DistilBERT...")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=LABEL_NAMES,
            label2id={
                value: key
                for key, value
                in LABEL_NAMES.items()
            },
        )
    )

    # ========================================================
    # DATA COLLATOR
    # ========================================================

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_args = TrainingArguments(
        output_dir=str(
            CHECKPOINT_DIR
        ),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        warmup_steps=500,

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="steps",

        logging_steps=50,

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",

        greater_is_better=True,

        save_total_limit=2,

        fp16=torch.cuda.is_available(),

        seed=SEED,

        data_seed=SEED,

        dataloader_num_workers=0,

        report_to="none",

        push_to_hub=False,

        remove_unused_columns=True,
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING ROBUST TRAINING")
    print("=" * 70)

    trainer.train()

    # ========================================================
    # CLEAN VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("CLEAN VALIDATION")
    print("=" * 70)

    validation_metrics, _ = evaluate_dataset(
        trainer,
        val_dataset,
        val_df,
        "validation",
        "validation",
    )

    # ========================================================
    # CLEAN TEST
    # ========================================================

    print()
    print("=" * 70)
    print("CLEAN TEST")
    print("=" * 70)

    test_metrics, _ = evaluate_dataset(
        trainer,
        test_dataset,
        test_df,
        "test",
        "test",
    )

    # ========================================================
    # ROBUSTNESS TEST
    # ========================================================

    robustness_df = (
        load_robustness_dataset()
    )

    robustness_summary = (
        evaluate_robustness(
            trainer,
            tokenizer,
            robustness_df,
        )
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    final_model_dir = (
        OUTPUT_DIR
        / "best_model"
    )

    trainer.save_model(
        str(final_model_dir)
    )

    tokenizer.save_pretrained(
        str(final_model_dir)
    )

    print()
    print(
        f"Best model saved:\n"
        f"{final_model_dir}"
    )

    # ========================================================
    # FINAL EXPERIMENT SUMMARY
    # ========================================================

    experiment = {

        "experiment": "Experiment 5",

        "name": (
            "Proposed Robust DistilBERT Model"
        ),

        "model": MODEL_NAME,

        "task": (
            "aspect_term_sentiment_classification"
        ),

        "seed": SEED,

        "max_length": MAX_LENGTH,

        "train_batch_size": (
            TRAIN_BATCH_SIZE
        ),

        "eval_batch_size": (
            EVAL_BATCH_SIZE
        ),

        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),

        "effective_batch_size": (
            TRAIN_BATCH_SIZE
            *
            GRADIENT_ACCUMULATION_STEPS
        ),

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "max_epochs": NUM_EPOCHS,

        "fp16": torch.cuda.is_available(),

        "original_train_instances": int(
            len(train_df)
        ),

        "augmented_train_instances": int(
            len(robust_train_df)
        ),

        "validation_instances": int(
            len(val_df)
        ),

        "test_instances": int(
            len(test_df)
        ),

        "robustness_instances": int(
            len(robustness_df)
        ),

        "augmentation_ratio": (
            AUGMENTATION_RATIO
        ),

        "validation": validation_metrics,

        "test": test_metrics,

        "robustness": robustness_summary,
    }

    metrics_output = (
        OUTPUT_DIR
        / "metrics.json"
    )

    with open(
        metrics_output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            experiment,
            f,
            indent=4,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print("Generated files:")

    generated_files = [
        "metrics.json",
        "robust_training_data.csv",
        "validation_predictions.csv",
        "test_predictions.csv",
        "robustness_original_predictions.csv",
        "robustness_predictions.csv",
        "robustness_by_perturbation_type.csv",
        "robustness_summary.json",
    ]

    for filename in generated_files:

        print(
            f" - {filename}"
        )

    print(
        "\nModel:"
    )

    print(
        f" - {final_model_dir}"
    )


if __name__ == "__main__":
    main()