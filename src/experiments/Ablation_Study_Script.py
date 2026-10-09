from __future__ import annotations

import json
import random
import re
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
# EXPERIMENT 6
# ABLATION STUDIES
#
# Purpose
# -------
# Determine the contribution of different perturbation
# families used during robust training.
#
# A0: Clean-only DistilBERT
#
# A1: Clean + context insertion
#
# A2: Clean + neutral/noise perturbations
#     - neutral_suffix
#     - irrelevant_noise
#
# A3: Clean + surface perturbations
#     - punctuation_variation
#     - whitespace_variation
#
# A4: Clean + semantic/context perturbations
#     - context_insertion
#     - neutral_suffix
#     - irrelevant_noise
#     - intensifier_insertion
#     - diminisher_insertion
#
# A5: Clean + ALL seven perturbation families
#     - context_insertion
#     - neutral_suffix
#     - intensifier_insertion
#     - diminisher_insertion
#     - irrelevant_noise
#     - punctuation_variation
#     - whitespace_variation
#
# IMPORTANT
# ---------
# The existing robustness dataset is NEVER used for training.
# It remains an evaluation-only dataset.
#
# All ablations use:
#     - same model
#     - same seed
#     - same train/validation/test split
#     - same optimizer settings
#     - same number of epochs
#     - same robustness evaluation set
#
# This isolates the effect of the training perturbations.
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

EARLY_STOPPING_PATIENCE = 2

WARMUP_STEPS = 500

AUGMENTATIONS_PER_INSTANCE = 1

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

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
    / "ablation_studies"
)

CHECKPOINT_ROOT = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ABLATION DEFINITIONS
# ============================================================

ABLATIONS = {

    "A0_clean_only": {
        "description": (
            "Original clean training data only."
        ),
        "perturbations": [],
    },

    "A1_context": {
        "description": (
            "Clean data + context insertion."
        ),
        "perturbations": [
            "context_insertion",
        ],
    },

    "A2_neutral_noise": {
        "description": (
            "Clean data + neutral suffix + "
            "irrelevant noise."
        ),
        "perturbations": [
            "neutral_suffix",
            "irrelevant_noise",
        ],
    },

    "A3_surface": {
        "description": (
            "Clean data + punctuation + "
            "whitespace variation."
        ),
        "perturbations": [
            "punctuation_variation",
            "whitespace_variation",
        ],
    },

    "A4_context_semantic": {
        "description": (
            "Clean data + context, neutral, noise, "
            "intensifier and diminisher perturbations."
        ),
        "perturbations": [
            "context_insertion",
            "neutral_suffix",
            "irrelevant_noise",
            "intensifier_insertion",
            "diminisher_insertion",
        ],
    },

    "A5_all_perturbations": {
        "description": (
            "Clean data + all seven controlled "
            "perturbation families."
        ),
        "perturbations": [
            "context_insertion",
            "neutral_suffix",
            "intensifier_insertion",
            "diminisher_insertion",
            "irrelevant_noise",
            "punctuation_variation",
            "whitespace_variation",
        ],
    },
}


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

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{split}: missing columns "
            f"{sorted(missing)}"
        )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    return df


# ============================================================
# PERTURBATION FUNCTIONS
# ============================================================

def perturb_context_insertion(
    sentence: str,
) -> str:

    prefixes = [
        "From the review, ",
        "In this review, ",
        "According to the customer, ",
        "In the customer's account, ",
    ]

    prefix = random.choice(
        prefixes
    )

    return prefix + sentence


def perturb_neutral_suffix(
    sentence: str,
) -> str:

    suffixes = [
        " The visit was on a weekday.",
        " The review describes a restaurant visit.",
        " This was part of the customer's overall experience.",
        " The customer included this detail in the review.",
    ]

    suffix = random.choice(
        suffixes
    )

    return (
        sentence.rstrip()
        + suffix
    )


def perturb_irrelevant_noise(
    sentence: str,
) -> str:

    variants = [
        " indeed",
        " overall",
        " as noted",
        " in particular",
    ]

    noise = random.choice(
        variants
    )

    stripped = sentence.rstrip()

    if stripped.endswith("."):

        return (
            stripped[:-1]
            + noise
            + "."
        )

    if stripped.endswith("!"):

        return (
            stripped[:-1]
            + noise
            + "!"
        )

    if stripped.endswith("?"):

        return (
            stripped[:-1]
            + noise
            + "?"
        )

    return (
        stripped
        + noise
    )


def perturb_punctuation(
    sentence: str,
) -> str:

    stripped = sentence.rstrip()

    if stripped.endswith("."):

        return (
            stripped[:-1]
            + "!"
        )

    if stripped.endswith("!"):

        return (
            stripped[:-1]
            + "."
        )

    if stripped.endswith("?"):

        return (
            stripped[:-1]
            + "."
        )

    return (
        stripped
        + "."
    )


def perturb_whitespace(
    sentence: str,
) -> str:

    words = sentence.split()

    if len(words) < 3:

        return sentence

    index = random.randint(
        1,
        len(words) - 1,
    )

    return (
        " ".join(words[:index])
        + "  "
        + " ".join(words[index:])
    )


def perturb_intensifier(
    sentence: str,
) -> str:

    candidate_words = [
        "good",
        "great",
        "excellent",
        "bad",
        "poor",
        "terrible",
        "amazing",
        "wonderful",
        "awful",
        "nice",
        "friendly",
        "slow",
        "fast",
        "delicious",
        "tasty",
        "disappointing",
        "helpful",
        "unhelpful",
    ]

    pattern = (
        r"\b("
        + "|".join(
            re.escape(word)
            for word in candidate_words
        )
        + r")\b"
    )

    match = re.search(
        pattern,
        sentence,
        flags=re.IGNORECASE,
    )

    if match is None:

        return sentence

    intensifier = random.choice(
        [
            "really",
            "quite",
            "especially",
        ]
    )

    original_word = match.group(0)

    replacement = (
        f"{intensifier} "
        f"{original_word}"
    )

    return (
        sentence[:match.start()]
        + replacement
        + sentence[match.end():]
    )


def perturb_diminisher(
    sentence: str,
) -> str:

    candidate_words = [
        "good",
        "great",
        "excellent",
        "bad",
        "poor",
        "terrible",
        "amazing",
        "wonderful",
        "awful",
        "nice",
        "friendly",
        "slow",
        "fast",
        "delicious",
        "tasty",
        "disappointing",
        "helpful",
        "unhelpful",
    ]

    pattern = (
        r"\b("
        + "|".join(
            re.escape(word)
            for word in candidate_words
        )
        + r")\b"
    )

    match = re.search(
        pattern,
        sentence,
        flags=re.IGNORECASE,
    )

    if match is None:

        return sentence

    diminisher = random.choice(
        [
            "somewhat",
            "relatively",
            "rather",
        ]
    )

    original_word = match.group(0)

    replacement = (
        f"{diminisher} "
        f"{original_word}"
    )

    return (
        sentence[:match.start()]
        + replacement
        + sentence[match.end():]
    )


PERTURBATION_FUNCTIONS = {

    "context_insertion":
        perturb_context_insertion,

    "neutral_suffix":
        perturb_neutral_suffix,

    "intensifier_insertion":
        perturb_intensifier,

    "diminisher_insertion":
        perturb_diminisher,

    "irrelevant_noise":
        perturb_irrelevant_noise,

    "punctuation_variation":
        perturb_punctuation,

    "whitespace_variation":
        perturb_whitespace,
}


# ============================================================
# GENERATE AUGMENTED DATA
# ============================================================

def generate_augmentation(
    train_df: pd.DataFrame,
    perturbation_types: list[str],
) -> pd.DataFrame:

    if not perturbation_types:

        return pd.DataFrame(
            columns=[
                "sentence_id",
                "sentence",
                "aspect",
                "polarity",
                "label",
                "training_source",
                "perturbation_type",
            ]
        )

    rows = []

    counter = 0

    for _, row in train_df.iterrows():

        original_sentence = str(
            row["sentence"]
        )

        for perturbation_type in (
            perturbation_types
        ):

            function = (
                PERTURBATION_FUNCTIONS[
                    perturbation_type
                ]
            )

            for _ in range(
                AUGMENTATIONS_PER_INSTANCE
            ):

                counter += 1

                perturbed_sentence = (
                    function(
                        original_sentence
                    )
                )

                if not perturbed_sentence:
                    continue

                perturbed_sentence = (
                    perturbed_sentence.strip()
                )

                if (
                    not perturbed_sentence
                    or
                    perturbed_sentence
                    ==
                    original_sentence
                ):
                    continue

                rows.append(
                    {
                        "sentence_id": (
                            f"ABLATION_TRAIN_"
                            f"{counter:08d}"
                        ),

                        "sentence": (
                            perturbed_sentence
                        ),

                        "aspect": str(
                            row["aspect"]
                        ),

                        "polarity": str(
                            row["polarity"]
                        ),

                        "label": int(
                            row["label"]
                        ),

                        "training_source": (
                            "perturbed"
                        ),

                        "perturbation_type": (
                            perturbation_type
                        ),
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# BUILD TRAINING DATA
# ============================================================

def build_training_dataframe(
    train_df: pd.DataFrame,
    perturbation_types: list[str],
) -> pd.DataFrame:

    clean_df = train_df[
        [
            "sentence_id",
            "sentence",
            "aspect",
            "polarity",
            "label",
        ]
    ].copy()

    clean_df[
        "training_source"
    ] = "clean"

    clean_df[
        "perturbation_type"
    ] = "none"

    augmented_df = (
        generate_augmentation(
            train_df,
            perturbation_types,
        )
    )

    combined = pd.concat(
        [
            clean_df,
            augmented_df,
        ],
        ignore_index=True,
    )

    combined = combined.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(
        drop=True
    )

    return combined


# ============================================================
# DATASET CONVERSION
# ============================================================

def dataframe_to_dataset(
    df: pd.DataFrame,
) -> Dataset:

    return Dataset.from_dict(
        {
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
    )


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

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[
            "sentence",
            "aspect",
        ],
        desc="Tokenizing",
    )


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
# GENERIC EVALUATION
# ============================================================

def evaluate_model(
    trainer: Trainer,
    dataset: Dataset,
    dataframe: pd.DataFrame,
):

    prediction_output = (
        trainer.predict(dataset)
    )

    logits = (
        prediction_output.predictions
    )

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

    report = classification_report(
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

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1, 2],
    )

    metrics = {
        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "classification_report": report,

        "confusion_matrix": (
            cm.tolist()
        ),
    }

    return (
        metrics,
        predictions,
    )


# ============================================================
# LOAD ROBUSTNESS DATASET
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

    df["label"] = (
        df["label"]
        .astype(int)
    )

    return df


# ============================================================
# ROBUSTNESS EVALUATION
# ============================================================

def evaluate_robustness(
    trainer: Trainer,
    tokenizer,
    robustness_df: pd.DataFrame,
):

    # --------------------------------------------------------
    # Original sentences
    # --------------------------------------------------------

    original_df = (
        robustness_df[
            [
                "sentence_id",
                "aspect",
                "true_polarity",
                "original_sentence",
                "label",
            ]
        ]
        .drop_duplicates(
            subset=[
                "sentence_id",
                "aspect",
            ]
        )
        .reset_index(drop=True)
    )

    original_input_df = (
        original_df.rename(
            columns={
                "original_sentence":
                    "sentence"
            }
        )
    )

    original_hf = (
        dataframe_to_dataset(
            original_input_df
        )
    )

    original_hf = tokenize_dataset(
        original_hf,
        tokenizer,
    )

    (
        original_metrics,
        original_predictions,
    ) = evaluate_model(
        trainer,
        original_hf,
        original_input_df,
    )

    original_df[
        "original_predicted_label"
    ] = original_predictions

    # --------------------------------------------------------
    # Perturbed sentences
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

    perturbed_df = (
        perturbed_df.rename(
            columns={
                "perturbed_sentence":
                    "sentence"
            }
        )
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
    ) = evaluate_model(
        trainer,
        perturbed_hf,
        perturbed_df,
    )

    perturbed_df[
        "predicted_label"
    ] = perturbed_predictions

    # --------------------------------------------------------
    # Attach original predictions
    # --------------------------------------------------------

    lookup = original_df[
        [
            "sentence_id",
            "aspect",
            "original_predicted_label",
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
        "original_correct"
    ] = (
        result[
            "original_predicted_label"
        ]
        ==
        result["label"]
    )

    result[
        "perturbed_correct"
    ] = (
        result[
            "predicted_label"
        ]
        ==
        result["label"]
    )

    result[
        "correct_to_incorrect"
    ] = (
        result["original_correct"]
        &
        ~result["perturbed_correct"]
    )

    result[
        "incorrect_to_correct"
    ] = (
        ~result["original_correct"]
        &
        result["perturbed_correct"]
    )

    # --------------------------------------------------------
    # Overall robustness
    # --------------------------------------------------------

    consistency = float(
        result[
            "prediction_consistent"
        ].mean()
    )

    flip_rate = float(
        result[
            "prediction_flip"
        ].mean()
    )

    correct_to_incorrect = float(
        result[
            "correct_to_incorrect"
        ].mean()
    )

    incorrect_to_correct = float(
        result[
            "incorrect_to_correct"
        ].mean()
    )

    # --------------------------------------------------------
    # By perturbation type
    # --------------------------------------------------------

    rows = []

    for (
        perturbation_type,
        group,
    ) in result.groupby(
        "perturbation_type",
        sort=True,
    ):

        rows.append(
            {
                "perturbation_type":
                    perturbation_type,

                "instances":
                    int(len(group)),

                "accuracy":
                    float(
                        group[
                            "perturbed_correct"
                        ].mean()
                    ),

                "prediction_consistency":
                    float(
                        group[
                            "prediction_consistent"
                        ].mean()
                    ),

                "prediction_flip_rate":
                    float(
                        group[
                            "prediction_flip"
                        ].mean()
                    ),

                "correct_to_incorrect_rate":
                    float(
                        group[
                            "correct_to_incorrect"
                        ].mean()
                    ),

                "incorrect_to_correct_rate":
                    float(
                        group[
                            "incorrect_to_correct"
                        ].mean()
                    ),
            }
        )

    by_type = pd.DataFrame(
        rows
    )

    return {
        "original_metrics":
            original_metrics,

        "perturbed_metrics":
            perturbed_metrics,

        "prediction_consistency":
            consistency,

        "prediction_flip_rate":
            flip_rate,

        "correct_to_incorrect_rate":
            correct_to_incorrect,

        "incorrect_to_correct_rate":
            incorrect_to_correct,

        "by_perturbation_type":
            by_type,

        "detailed_results":
            result,
    }


# ============================================================
# CREATE TRAINER
# ============================================================

def create_trainer(
    train_dataset,
    val_dataset,
    tokenizer,
    checkpoint_dir: Path,
):

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

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )
    )

    training_args = TrainingArguments(
        output_dir=str(
            checkpoint_dir
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

        warmup_steps=WARMUP_STEPS,

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
                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                )
            )
        ],
    )

    return trainer


# ============================================================
# RUN ONE ABLATION
# ============================================================

def run_ablation(
    ablation_name: str,
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    tokenizer,
):

    print()
    print("=" * 80)
    print(
        f"RUNNING {ablation_name}"
    )
    print("=" * 80)

    print(
        f"Description:\n"
        f"{config['description']}"
    )

    print()
    print(
        "Perturbations:"
    )

    if config["perturbations"]:

        for perturbation in (
            config["perturbations"]
        ):

            print(
                f" - {perturbation}"
            )

    else:

        print(
            " - None"
        )

    # --------------------------------------------------------
    # Build training data
    # --------------------------------------------------------

    robust_train_df = (
        build_training_dataframe(
            train_df,
            config["perturbations"],
        )
    )

    print()
    print(
        f"Clean training instances     : "
        f"{len(train_df):,}"
    )

    print(
        f"Perturbed training instances : "
        f"{len(robust_train_df) - len(train_df):,}"
    )

    print(
        f"Total training instances     : "
        f"{len(robust_train_df):,}"
    )

    # --------------------------------------------------------
    # Save training manifest
    # --------------------------------------------------------

    ablation_dir = (
        OUTPUT_DIR
        / ablation_name
    )

    ablation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_manifest = (
        ablation_dir
        / "training_data.csv"
    )

    robust_train_df.to_csv(
        training_manifest,
        index=False,
    )

    # --------------------------------------------------------
    # Build HF datasets
    # --------------------------------------------------------

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

    train_dataset = (
        tokenize_dataset(
            train_dataset,
            tokenizer,
        )
    )

    val_dataset = (
        tokenize_dataset(
            val_dataset,
            tokenizer,
        )
    )

    test_dataset = (
        tokenize_dataset(
            test_dataset,
            tokenizer,
        )
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    checkpoint_dir = (
        CHECKPOINT_ROOT
        / ablation_name
    )

    trainer = create_trainer(
        train_dataset,
        val_dataset,
        tokenizer,
        checkpoint_dir,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print(
        f"Starting training: "
        f"{ablation_name}"
    )

    train_output = (
        trainer.train()
    )

    # --------------------------------------------------------
    # Clean validation
    # --------------------------------------------------------

    validation_metrics, _ = (
        evaluate_model(
            trainer,
            val_dataset,
            val_df,
        )
    )

    # --------------------------------------------------------
    # Clean test
    # --------------------------------------------------------

    test_metrics, test_predictions = (
        evaluate_model(
            trainer,
            test_dataset,
            test_df,
        )
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    test_predictions_df = (
        test_df.copy()
    )

    test_predictions_df[
        "predicted_label"
    ] = test_predictions

    test_predictions_df[
        "predicted_polarity"
    ] = (
        test_predictions_df[
            "predicted_label"
        ].map(LABEL_NAMES)
    )

    test_predictions_df[
        "correct"
    ] = (
        test_predictions_df[
            "label"
        ]
        ==
        test_predictions_df[
            "predicted_label"
        ]
    )

    test_predictions_df.to_csv(
        ablation_dir
        / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Robustness evaluation
    # --------------------------------------------------------

    robustness = (
        evaluate_robustness(
            trainer,
            tokenizer,
            robustness_df,
        )
    )

    robustness[
        "by_perturbation_type"
    ].to_csv(
        ablation_dir
        / "robustness_by_perturbation_type.csv",
        index=False,
    )

    robustness[
        "detailed_results"
    ].to_csv(
        ablation_dir
        / "robustness_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {

        "experiment":
            "Experiment 6 - Ablation Study",

        "ablation":
            ablation_name,

        "description":
            config["description"],

        "model":
            MODEL_NAME,

        "seed":
            SEED,

        "perturbations":
            config["perturbations"],

        "clean_training_instances":
            int(len(train_df)),

        "perturbed_training_instances":
            int(
                len(robust_train_df)
                - len(train_df)
            ),

        "total_training_instances":
            int(
                len(robust_train_df)
            ),

        "validation":
            validation_metrics,

        "test":
            test_metrics,

        "robustness": {

            "original":
                robustness[
                    "original_metrics"
                ],

            "perturbed":
                robustness[
                    "perturbed_metrics"
                ],

            "prediction_consistency":
                robustness[
                    "prediction_consistency"
                ],

            "prediction_flip_rate":
                robustness[
                    "prediction_flip_rate"
                ],

            "correct_to_incorrect_rate":
                robustness[
                    "correct_to_incorrect_rate"
                ],

            "incorrect_to_correct_rate":
                robustness[
                    "incorrect_to_correct_rate"
                ],
        },

        "training":
            {
                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "max_epochs":
                    NUM_EPOCHS,

                "train_batch_size":
                    TRAIN_BATCH_SIZE,

                "eval_batch_size":
                    EVAL_BATCH_SIZE,

                "gradient_accumulation_steps":
                    GRADIENT_ACCUMULATION_STEPS,

                "effective_batch_size":
                    (
                        TRAIN_BATCH_SIZE
                        *
                        GRADIENT_ACCUMULATION_STEPS
                    ),

                "max_length":
                    MAX_LENGTH,

                "fp16":
                    torch.cuda.is_available(),
            },
    }

    with open(
        ablation_dir
        / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    final_model_dir = (
        ablation_dir
        / "best_model"
    )

    trainer.save_model(
        str(final_model_dir)
    )

    tokenizer.save_pretrained(
        str(final_model_dir)
    )

    # --------------------------------------------------------
    # Release model memory
    # --------------------------------------------------------

    del trainer

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return summary


# ============================================================
# BUILD COMPARATIVE ABLATION TABLE
# ============================================================

def build_comparative_table(
    summaries: list[dict],
):

    rows = []

    for summary in summaries:

        test = summary["test"]

        robustness = (
            summary["robustness"]
        )

        rows.append(
            {
                "ablation":
                    summary["ablation"],

                "description":
                    summary["description"],

                "perturbations":
                    ", ".join(
                        summary[
                            "perturbations"
                        ]
                    )
                    if summary[
                        "perturbations"
                    ]
                    else "none",

                "clean_train_instances":
                    summary[
                        "clean_training_instances"
                    ],

                "perturbed_train_instances":
                    summary[
                        "perturbed_training_instances"
                    ],

                "total_train_instances":
                    summary[
                        "total_training_instances"
                    ],

                "test_accuracy":
                    test[
                        "accuracy"
                    ],

                "test_macro_f1":
                    test[
                        "macro_f1"
                    ],

                "test_weighted_f1":
                    test[
                        "weighted_f1"
                    ],

                "robust_accuracy":
                    robustness[
                        "perturbed"
                    ][
                        "accuracy"
                    ],

                "robust_macro_f1":
                    robustness[
                        "perturbed"
                    ][
                        "macro_f1"
                    ],

                "prediction_consistency":
                    robustness[
                        "prediction_consistency"
                    ],

                "prediction_flip_rate":
                    robustness[
                        "prediction_flip_rate"
                    ],

                "correct_to_incorrect_rate":
                    robustness[
                        "correct_to_incorrect_rate"
                    ],

                "incorrect_to_correct_rate":
                    robustness[
                        "incorrect_to_correct_rate"
                    ],
            }
        )

    table = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Add differences relative to A0
    # --------------------------------------------------------

    baseline_rows = table[
        table["ablation"]
        ==
        "A0_clean_only"
    ]

    if len(baseline_rows) == 1:

        baseline = (
            baseline_rows.iloc[0]
        )

        table[
            "delta_test_accuracy"
        ] = (
            table[
                "test_accuracy"
            ]
            - baseline[
                "test_accuracy"
            ]
        )

        table[
            "delta_test_macro_f1"
        ] = (
            table[
                "test_macro_f1"
            ]
            - baseline[
                "test_macro_f1"
            ]
        )

        table[
            "delta_robust_accuracy"
        ] = (
            table[
                "robust_accuracy"
            ]
            - baseline[
                "robust_accuracy"
            ]
        )

        table[
            "delta_robust_macro_f1"
        ] = (
            table[
                "robust_macro_f1"
            ]
            - baseline[
                "robust_macro_f1"
            ]
        )

        table[
            "delta_consistency"
        ] = (
            table[
                "prediction_consistency"
            ]
            - baseline[
                "prediction_consistency"
            ]
        )

        table[
            "delta_flip_rate"
        ] = (
            table[
                "prediction_flip_rate"
            ]
            - baseline[
                "prediction_flip_rate"
            ]
        )

    return table


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything()

    print("=" * 80)
    print("EXPERIMENT 6")
    print("ABLATION STUDIES")
    print("=" * 80)

    print()
    print(
        f"Model        : {MODEL_NAME}"
    )

    print(
        f"Seed         : {SEED}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Epochs       : {NUM_EPOCHS}"
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

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df = load_split(
        "train"
    )

    val_df = load_split(
        "val"
    )

    test_df = load_split(
        "test"
    )

    robustness_df = (
        load_robustness_dataset()
    )

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        f"Train       : {len(train_df):,}"
    )

    print(
        f"Validation  : {len(val_df):,}"
    )

    print(
        f"Test        : {len(test_df):,}"
    )

    print(
        f"Robustness  : {len(robustness_df):,}"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print()
    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    # ========================================================
    # RUN ALL ABLATIONS
    # ========================================================

    summaries = []

    total_ablations = len(
        ABLATIONS
    )

    for counter, (
        ablation_name,
        config,
    ) in enumerate(
        ABLATIONS.items(),
        start=1,
    ):

        print()
        print(
            "#" * 80
        )

        print(
            f"ABLATION "
            f"{counter}/{total_ablations}"
        )

        print(
            f"{ablation_name}"
        )

        print(
            "#" * 80
        )

        seed_everything(
            SEED
        )

        summary = run_ablation(
            ablation_name,
            config,
            train_df,
            val_df,
            test_df,
            robustness_df,
            tokenizer,
        )

        summaries.append(
            summary
        )

    # ========================================================
    # COMPARATIVE TABLE
    # ========================================================

    print()
    print("=" * 80)
    print("BUILDING ABLATION COMPARISON")
    print("=" * 80)

    comparison = (
        build_comparative_table(
            summaries
        )
    )

    comparison_file = (
        OUTPUT_DIR
        / "ablation_results.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    print()
    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved:\n"
        f"{comparison_file}"
    )

    # ========================================================
    # JSON SUMMARY
    # ========================================================

    all_results = {
        "experiment":
            "Experiment 6 - Ablation Study",

        "model":
            MODEL_NAME,

        "seed":
            SEED,

        "ablation_definitions":
            ABLATIONS,

        "results":
            summaries,
    }

    json_file = (
        OUTPUT_DIR
        / "ablation_results.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
        )

    print()
    print(
        f"Saved:\n"
        f"{json_file}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("EXPERIMENT 6 COMPLETE")
    print("=" * 80)

    print()
    print(
        "Generated ablation directories:"
    )

    for ablation_name in ABLATIONS:

        print(
            f" - "
            f"{OUTPUT_DIR / ablation_name}"
        )

    print()
    print(
        "Main comparison:"
    )

    print(
        f" - {comparison_file}"
    )

    print()
    print(
        "JSON summary:"
    )

    print(
        f" - {json_file}"
    )

    print()
    print(
        "The ablation results can now be used "
        "to construct the final comparative table."
    )


if __name__ == "__main__":
    main()
