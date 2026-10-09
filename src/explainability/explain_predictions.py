from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MODEL_DIR = Path(
    "results/transformer/distilbert_baseline/best_model"
)

PREDICTIONS_FILE = Path(
    "results/transformer/distilbert_baseline/test_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/transformer/distilbert_baseline/explainability"
)

# Start small for Experiment 1.
# Change to None after the pipeline is verified.
MAX_EXAMPLES = None

MAX_LENGTH = 128

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# SETUP
# ============================================================

def main():

    print("=" * 70)
    print("SHAP EXPLAINABILITY — EXPERIMENT 1")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.random.seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device : {device}")
    print(f"SHAP   : {shap.__version__}")

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    print()
    print("Loading predictions...")

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    print(
        f"Total test instances: {len(df):,}"
    )

    # Keep a reproducible sample for Experiment 1.
    if MAX_EXAMPLES is not None:
        df = df.head(MAX_EXAMPLES).copy()

    print(
        f"Explaining instances: {len(df):,}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(device)
    model.eval()

    print(
        f"Model: {type(model).__name__}"
    )

    # --------------------------------------------------------
    # Prediction function for SHAP
    # --------------------------------------------------------

    # --------------------------------------------------------
    # SHAP-compatible serialized ABSA input
    # --------------------------------------------------------

   
    # --------------------------------------------------------
    # Prediction function for SHAP
    # --------------------------------------------------------

        # --------------------------------------------------------
    # SHAP EXPLANATION
    #
    # Experiment 1:
    # Keep the aspect fixed.
    # Explain the contribution of sentence tokens.
    # --------------------------------------------------------

    def predict_sentence(texts, fixed_aspect):

        aspects = [
            fixed_aspect
            for _ in texts
        ]

        sentences = [
            str(text)
            for text in texts
        ]

        encoded = tokenizer(
            aspects,
            sentences,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = model(
                **encoded
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        return probabilities.cpu().numpy()


    

    # --------------------------------------------------------
    # Explain examples
    # --------------------------------------------------------

    records = []

    examples = []

    print()
    print("=" * 70)
    print("GENERATING EXPLANATIONS")
    print("=" * 70)

    for index, row in df.iterrows():

        aspect = str(
            row["aspect"]
        )

        sentence = str(
            row["sentence"]
        )
        
        true_label = int(
            row["true_label"]
        )

        predicted_label = int(
            row["predicted_label"]
        )

        print(
            f"[{index + 1}/{len(df)}] "
            f"{row['sentence_id']} | "
            f"aspect='{aspect}' | "
            f"prediction={LABEL_NAMES[predicted_label]}"
        )

        # ----------------------------------------------------
        # SHAP explainer for this aspect
        # ----------------------------------------------------

        def model_for_shap(texts):

            return predict_sentence(
                texts,
                fixed_aspect=aspect,
            )

        masker = shap.maskers.Text(
            tokenizer=tokenizer
        )

        explainer = shap.Explainer(
            model_for_shap,
            masker,
            output_names=[
                LABEL_NAMES[0],
                LABEL_NAMES[1],
                LABEL_NAMES[2],
            ],
        )

        shap_values = explainer(
            [sentence]
        )

        # ----------------------------------------------------
        # Extract tokens and SHAP values
        # ----------------------------------------------------

        tokens = shap_values.data[0]

        values = np.asarray(
            shap_values.values[0]
        )

        print(
            f"SHAP shape: {values.shape}"
        )

        if values.ndim != 2:
            raise RuntimeError(
                "Unexpected SHAP value shape: "
                f"{values.shape}"
            )

        predicted_values = values[
            :,
            predicted_label
        ]

        # ----------------------------------------------------
        # Store token-level explanations
        # ----------------------------------------------------

        token_records = []

        for token_index, token in enumerate(tokens):

            token = str(token)

            class_values = {
                LABEL_NAMES[class_id]:
                    float(
                        values[
                            token_index,
                            class_id
                        ]
                    )
                for class_id in range(3)
            }

            record = {
                "sentence_id": row["sentence_id"],
                "aspect": aspect,
                "sentence": sentence,
                "true_polarity": row["true_polarity"],
                "predicted_polarity": row["predicted_polarity"],
                "correct": bool(row["correct"]),
                "token_index": token_index,
                "token": token,
                "shap_negative": class_values["negative"],
                "shap_neutral": class_values["neutral"],
                "shap_positive": class_values["positive"],
                "shap_predicted_class": float(
                    predicted_values[token_index]
                ),
            }

            token_records.append(record)
            records.append(record)

        # ----------------------------------------------------
        # Top positive/negative evidence
        # ----------------------------------------------------

        ranked = sorted(
            token_records,
            key=lambda x: abs(
                x["shap_predicted_class"]
            ),
            reverse=True,
        )

        examples.append(
            {
                "sentence_id": row["sentence_id"],
                "aspect": aspect,
                "sentence": sentence,
                "true_polarity": row["true_polarity"],
                "predicted_polarity": row[
                    "predicted_polarity"
                ],
                "correct": bool(
                    row["correct"]
                ),
                "top_evidence": ranked[:10],
            }
        )

    # --------------------------------------------------------
    # Save token-level results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        records
    )

    results_file = (
        OUTPUT_DIR
        / "explanation_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save examples
    # --------------------------------------------------------

    examples_file = (
        OUTPUT_DIR
        / "explanation_examples.json"
    )

    with open(
        examples_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            examples,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment": "SHAP token attribution",
        "model": str(MODEL_DIR),
        "seed": SEED,
        "device": str(device),
        "shap_version": shap.__version__,
        "max_length": MAX_LENGTH,
        "instances_explained": len(df),
        "total_token_attributions": len(
            results_df
        ),
        "correct_instances": int(
            df["correct"].sum()
        ),
        "incorrect_instances": int(
            (~df["correct"]).sum()
        ),
    }

    summary_file = (
        OUTPUT_DIR
        / "explanation_summary.json"
    )

    with open(
        summary_file,
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
    print("EXPLAINABILITY EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Results : {OUTPUT_DIR}"
    )

    print(
        f"Token explanations : {results_file}"
    )

    print(
        f"Examples           : {examples_file}"
    )

    print(
        f"Summary            : {summary_file}"
    )


if __name__ == "__main__":
    main()