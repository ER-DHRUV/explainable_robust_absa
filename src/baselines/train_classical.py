from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


# ============================================================
# Configuration
# ============================================================

SEED = 42

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "baselines"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Dataset loading
# ============================================================

def load_split(split: str) -> pd.DataFrame:

    path = DATA_DIR / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "sentence_id",
        "sentence",
        "aspect",
        "polarity",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{split}: missing columns: {sorted(missing)}"
        )

    polarity_to_label = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    df["label"] = df["polarity"].map(polarity_to_label)

    if df["label"].isna().any():
        unknown = df.loc[
            df["label"].isna(),
            "polarity"
        ].unique()

        raise ValueError(
            f"{split}: unknown polarity values: {unknown}"
        )

    df["label"] = df["label"].astype(int)

    return df


# ============================================================
# Target-aware text representation
# ============================================================

def build_target_aware_text(df: pd.DataFrame) -> pd.Series:
    """
    Construct the ATSA input.

    Missing aspect/sentence values are replaced with
    an empty string so TF-IDF receives valid text.

    Example:
    [ASPECT] service [CONTEXT] sentence text
    """

    aspect = df["aspect"].fillna("").astype(str)
    sentence = df["sentence"].fillna("").astype(str)

    return (
        "[ASPECT] "
        + aspect
        + " [CONTEXT] "
        + sentence
    )



# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
) -> dict:

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    for index, label_id in enumerate([0, 1, 2]):

        label_name = LABEL_NAMES[label_id]

        metrics[f"{label_name}_precision"] = float(
            precision[index]
        )

        metrics[f"{label_name}_recall"] = float(
            recall[index]
        )

        metrics[f"{label_name}_f1"] = float(
            f1[index]
        )

        metrics[f"{label_name}_support"] = int(
            support[index]
        )

    return metrics


def print_metrics(
    model_name: str,
    split_name: str,
    y_true,
    y_pred,
) -> dict:

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print()
    print("=" * 70)
    print(f"{model_name} — {split_name.upper()}")
    print("=" * 70)

    print(
        f"Accuracy   : {metrics['accuracy']:.4f}"
    )

    print(
        f"Macro-F1   : {metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted-F1: {metrics['weighted_f1']:.4f}"
    )

    print()
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[
                LABEL_NAMES[0],
                LABEL_NAMES[1],
                LABEL_NAMES[2],
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2],
        )
    )

    return metrics


# ============================================================
# Save experiment
# ============================================================

def save_experiment(
    model_name: str,
    model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_pred,
    test_pred,
    val_metrics: dict,
    test_metrics: dict,
) -> None:

    model_dir = RESULTS_DIR / model_name
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    val_output = val_df[
        [
            "sentence_id",
            "sentence",
            "aspect",
            "polarity",
        ]
    ].copy()

    val_output["true_label"] = val_df["label"].values

    val_output["predicted_label"] = val_pred

    val_output["true_polarity"] = (
        val_output["true_label"]
        .map(LABEL_NAMES)
    )

    val_output["predicted_polarity"] = (
        val_output["predicted_label"]
        .map(LABEL_NAMES)
    )

    val_output.to_csv(
        model_dir / "validation_predictions.csv",
        index=False,
    )

    test_output = test_df[
        [
            "sentence_id",
            "sentence",
            "aspect",
            "polarity",
        ]
    ].copy()

    test_output["true_label"] = test_df["label"].values

    test_output["predicted_label"] = test_pred

    test_output["true_polarity"] = (
        test_output["true_label"]
        .map(LABEL_NAMES)
    )

    test_output["predicted_polarity"] = (
        test_output["predicted_label"]
        .map(LABEL_NAMES)
    )

    test_output.to_csv(
        model_dir / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    results = {
        "model": model_name,
        "seed": SEED,
        "train_instances": len(train_df),
        "validation_instances": len(val_df),
        "test_instances": len(test_df),
        "validation": val_metrics,
        "test": test_metrics,
    }

    with open(
        model_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        f"\nSaved experiment results to: {model_dir}"
    )


# ============================================================
# Build TF-IDF features
# ============================================================

def build_vectorizer() -> FeatureUnion:

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=100_000,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )

    return FeatureUnion(
        [
            (
                "word_tfidf",
                word_vectorizer,
            ),
            (
                "char_tfidf",
                char_vectorizer,
            ),
        ]
    )


# ============================================================
# Majority baseline
# ============================================================

def run_majority_baseline(
    train_df,
    val_df,
    test_df,
):

    model_name = "majority"

    majority_label = (
        train_df["label"]
        .value_counts()
        .idxmax()
    )

    print()
    print("=" * 70)
    print("MAJORITY CLASS BASELINE")
    print("=" * 70)

    print(
        "Training-set majority label:",
        LABEL_NAMES[majority_label],
    )

    val_pred = np.full(
        len(val_df),
        majority_label,
    )

    test_pred = np.full(
        len(test_df),
        majority_label,
    )

    val_metrics = print_metrics(
        model_name,
        "validation",
        val_df["label"],
        val_pred,
    )

    test_metrics = print_metrics(
        model_name,
        "test",
        test_df["label"],
        test_pred,
    )

    save_experiment(
        model_name,
        None,
        train_df,
        val_df,
        test_df,
        val_pred,
        test_pred,
        val_metrics,
        test_metrics,
    )


# ============================================================
# Logistic Regression
# ============================================================

def run_logistic_regression(
    train_df,
    val_df,
    test_df,
):

    model_name = "tfidf_logistic_regression"

    print()
    print("=" * 70)
    print("TF-IDF + LOGISTIC REGRESSION")
    print("=" * 70)

    train_text = build_target_aware_text(train_df)
    val_text = build_target_aware_text(val_df)
    test_text = build_target_aware_text(test_df)

    vectorizer = build_vectorizer()

    model = Pipeline(
        [
            (
                "tfidf",
                vectorizer,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    C=2.0,
                    class_weight=None,
                    random_state=SEED,
                    solver="lbfgs",
                    
                ),
            ),
        ]
    )

    print("Training...")

    model.fit(
        train_text,
        train_df["label"],
    )

    print("Training complete.")

    val_pred = model.predict(val_text)
    test_pred = model.predict(test_text)

    val_metrics = print_metrics(
        model_name,
        "validation",
        val_df["label"],
        val_pred,
    )

    test_metrics = print_metrics(
        model_name,
        "test",
        test_df["label"],
        test_pred,
    )

    save_experiment(
        model_name,
        model,
        train_df,
        val_df,
        test_df,
        val_pred,
        test_pred,
        val_metrics,
        test_metrics,
    )


# ============================================================
# Linear SVM
# ============================================================

def run_linear_svm(
    train_df,
    val_df,
    test_df,
):

    model_name = "tfidf_linear_svm"

    print()
    print("=" * 70)
    print("TF-IDF + LINEAR SVM")
    print("=" * 70)

    train_text = build_target_aware_text(train_df)
    val_text = build_target_aware_text(val_df)
    test_text = build_target_aware_text(test_df)

    vectorizer = build_vectorizer()

    model = Pipeline(
        [
            (
                "tfidf",
                vectorizer,
            ),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight=None,
                    random_state=SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )

    print("Training...")

    model.fit(
        train_text,
        train_df["label"],
    )

    print("Training complete.")

    val_pred = model.predict(val_text)
    test_pred = model.predict(test_text)

    val_metrics = print_metrics(
        model_name,
        "validation",
        val_df["label"],
        val_pred,
    )

    test_metrics = print_metrics(
        model_name,
        "test",
        test_df["label"],
        test_pred,
    )

    save_experiment(
        model_name,
        model,
        train_df,
        val_df,
        test_df,
        val_pred,
        test_pred,
        val_metrics,
        test_metrics,
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed()

    print("=" * 70)
    print("CLASSICAL ABSA BASELINES")
    print("=" * 70)

    print(f"Random seed: {SEED}")

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    print()
    print("Dataset sizes:")
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
    # Baseline 0
    # --------------------------------------------------------

    run_majority_baseline(
        train_df,
        val_df,
        test_df,
    )

    # --------------------------------------------------------
    # Baseline 1
    # --------------------------------------------------------

    run_logistic_regression(
        train_df,
        val_df,
        test_df,
    )

    # --------------------------------------------------------
    # Baseline 2
    # --------------------------------------------------------

    run_linear_svm(
        train_df,
        val_df,
        test_df,
    )

    print()
    print("=" * 70)
    print("ALL CLASSICAL BASELINES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()