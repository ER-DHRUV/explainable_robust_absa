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

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "transformer"
    / "distilbert_baseline"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

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
            f"Dataset not found: {path}"
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

    # Explicitly enforce integer labels.
    df["label"] = df["label"].astype(int)

    return df


# ============================================================
# HUGGING FACE DATASET
# ============================================================

def dataframe_to_dataset(
    df: pd.DataFrame,
) -> Dataset:

    data = {
        "sentence": df["sentence"].astype(str).tolist(),
        "aspect": df["aspect"].astype(str).tolist(),
        "labels": df["label"].astype(int).tolist(),
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

def compute_metrics(eval_prediction):

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
# DETAILED EVALUATION
# ============================================================

def detailed_evaluation(
    trainer: Trainer,
    dataset: Dataset,
    dataframe: pd.DataFrame,
    split_name: str,
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
        f"DISTILBERT — {split_name.upper()}"
    )
    print("=" * 70)

    print(
        f"Accuracy   : {accuracy:.4f}"
    )

    print(
        f"Macro-F1   : {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1: {weighted_f1:.4f}"
    )

    print()

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

    print(report)

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1, 2],
    )

    print("Confusion matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    output = dataframe[
        [
            "sentence_id",
            "sentence",
            "aspect",
            "polarity",
        ]
    ].copy()

    output["true_label"] = labels

    output["predicted_label"] = predictions

    output["true_polarity"] = (
        output["true_label"]
        .map(LABEL_NAMES)
    )

    output["predicted_polarity"] = (
        output["predicted_label"]
        .map(LABEL_NAMES)
    )

    output["correct"] = (
        output["true_label"]
        == output["predicted_label"]
    )

    output.to_csv(
        OUTPUT_DIR
        / f"{split_name}_predictions.csv",
        index=False,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": classification_report(
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
        ),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything()

    print("=" * 70)
    print("DISTILBERT ABSA BASELINE")
    print("=" * 70)

    print(
        f"Model       : {MODEL_NAME}"
    )

    print(
        f"Max length  : {MAX_LENGTH}"
    )

    print(
        f"Batch size  : {TRAIN_BATCH_SIZE}"
    )

    print(
        f"Accumulation: {GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Effective BS: "
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
        f"CUDA         : {torch.cuda.is_available()}"
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

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    print()
    print("Dataset:")
    print(
        f"Train: {len(train_df):,}"
    )
    print(
        f"Val  : {len(val_df):,}"
    )
    print(
        f"Test : {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Convert to HF datasets
    # --------------------------------------------------------

    train_dataset = dataframe_to_dataset(
        train_df
    )

    val_dataset = dataframe_to_dataset(
        val_df
    )

    test_dataset = dataframe_to_dataset(
        test_df
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=LABEL_NAMES,
        label2id={
            value: key
            for key, value in LABEL_NAMES.items()
        },
    )

    # --------------------------------------------------------
    # Data collator
    # --------------------------------------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),

        # Training
        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        

        # Evaluation
        eval_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="steps",

        logging_steps=50,

        # Best checkpoint
        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",

        greater_is_better=True,

        save_total_limit=2,

        # Precision
        fp16=True,

        # Reproducibility
        seed=SEED,

        data_seed=SEED,

        # Performance
        dataloader_num_workers=0,

        report_to="none",

        # Avoid unnecessary external integrations
        push_to_hub=False,

        # Transformers 5.x
        remove_unused_columns=True,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    trainer.train()

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL VALIDATION EVALUATION")
    print("=" * 70)

    val_metrics = detailed_evaluation(
        trainer,
        val_dataset,
        val_df,
        "validation",
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_metrics = detailed_evaluation(
        trainer,
        test_dataset,
        test_df,
        "test",
    )

    # --------------------------------------------------------
    # Save configuration + metrics
    # --------------------------------------------------------

    experiment = {
        "model": MODEL_NAME,
        "task": "aspect_term_sentiment_classification",
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "effective_batch_size": (
            TRAIN_BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS
        ),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": NUM_EPOCHS,
        "fp16": True,
        "train_instances": len(train_df),
        "validation_instances": len(val_df),
        "test_instances": len(test_df),
        "validation": val_metrics,
        "test": test_metrics,
    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            experiment,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Save tokenizer + model
    # --------------------------------------------------------

    final_model_dir = (
        OUTPUT_DIR / "best_model"
    )

    trainer.save_model(
        str(final_model_dir)
    )

    tokenizer.save_pretrained(
        str(final_model_dir)
    )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Results: {OUTPUT_DIR}"
    )

    print(
        f"Best model: {final_model_dir}"
    )


if __name__ == "__main__":
    main()