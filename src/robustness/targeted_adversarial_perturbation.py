from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ROBUSTNESS EXPERIMENT 3
# TARGETED ADVERSARIAL PERTURBATION GENERATION
# ============================================================

SEED = 42

# Number of targeted perturbations generated per eligible
# original instance is controlled by the perturbation rules.
MAX_LENGTH = 128

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

INPUT_FILE = DATA_DIR / "test.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "transformer"
    / "distilbert_baseline"
    / "robustness"
    / "targeted_adversarial"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "targeted_adversarial_instances.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "targeted_adversarial_summary.json"
)


# ============================================================
# RANDOMNESS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# TARGETED PERTURBATION VOCABULARY
# ============================================================

NEGATION_PHRASES = [
    "not really",
    "not particularly",
    "not exactly",
]

INTENSIFIERS = [
    "very",
    "extremely",
    "especially",
]

DIMINISHERS = [
    "somewhat",
    "rather",
    "relatively",
]

CONTEXT_PREFIXES = [
    "In this review,",
    "According to the customer,",
    "From the review,",
    "For this visit,",
]

CONTEXT_SUFFIXES = [
    "This was part of the overall review.",
    "The customer mentioned this in the review.",
    "The visit was described in the review.",
]

NOISE_PHRASES = [
    "overall",
    "notably",
    "indeed",
    "in this case",
]


# ============================================================
# SENTIMENT LEXICON
#
# These are deliberately small, controlled vocabularies.
# The experiment is intended to test targeted sensitivity,
# not to perform unrestricted text generation.
# ============================================================

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "wonderful",
    "fantastic",
    "delicious",
    "perfect",
    "friendly",
    "pleasant",
    "fresh",
    "outstanding",
    "best",
    "nice",
    "love",
    "loved",
    "like",
    "liked",
    "enjoy",
    "enjoyed",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "horrible",
    "poor",
    "worst",
    "disappointing",
    "disappointed",
    "bland",
    "dry",
    "cold",
    "rude",
    "slow",
    "dirty",
    "expensive",
    "unpleasant",
    "hate",
    "hated",
    "dislike",
    "disliked",
}


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize whitespace for validation."""
    return re.sub(r"\s+", " ", str(text)).strip()


def contains_aspect(
    sentence: str,
    aspect: str,
) -> bool:

    sentence_norm = normalize_text(sentence).lower()
    aspect_norm = normalize_text(aspect).lower()

    if not aspect_norm:
        return False

    return aspect_norm in sentence_norm


def safe_insert_before(
    sentence: str,
    target: str,
    insertion: str,
) -> str | None:

    pattern = re.compile(
        re.escape(target),
        flags=re.IGNORECASE,
    )

    match = pattern.search(sentence)

    if not match:
        return None

    start = match.start()

    return (
        sentence[:start]
        + insertion
        + " "
        + sentence[start:]
    )


def safe_insert_after(
    sentence: str,
    target: str,
    insertion: str,
) -> str | None:

    pattern = re.compile(
        re.escape(target),
        flags=re.IGNORECASE,
    )

    match = pattern.search(sentence)

    if not match:
        return None

    end = match.end()

    return (
        sentence[:end]
        + " "
        + insertion
        + sentence[end:]
    )


def replace_first_word(
    sentence: str,
    word: str,
    replacement: str,
) -> str | None:

    pattern = re.compile(
        rf"\b{re.escape(word)}\b",
        flags=re.IGNORECASE,
    )

    if not pattern.search(sentence):
        return None

    return pattern.sub(
        replacement,
        sentence,
        count=1,
    )


def find_sentiment_words(
    sentence: str,
) -> list[str]:

    words = re.findall(
        r"\b[\w'-]+\b",
        sentence.lower(),
    )

    return [
        word
        for word in words
        if (
            word in POSITIVE_WORDS
            or word in NEGATIVE_WORDS
        )
    ]


# ============================================================
# PERTURBATION VALIDATION
# ============================================================

def validate_perturbation(
    original_sentence: str,
    perturbed_sentence: str,
    aspect: str,
) -> bool:

    if not perturbed_sentence:
        return False

    if (
        normalize_text(original_sentence)
        == normalize_text(perturbed_sentence)
    ):
        return False

    # The aspect must remain explicitly present.
    if not contains_aspect(
        perturbed_sentence,
        aspect,
    ):
        return False

    return True


# ============================================================
# PERTURBATION GENERATORS
# ============================================================

def generate_context_distraction(
    sentence: str,
    aspect: str,
) -> list[tuple[str, str]]:

    perturbations = []

    for prefix in CONTEXT_PREFIXES:

        perturbed = (
            f"{prefix} {sentence}"
        )

        perturbations.append(
            (
                "context_distraction",
                perturbed,
            )
        )

    for suffix in CONTEXT_SUFFIXES:

        perturbed = (
            f"{sentence} {suffix}"
        )

        perturbations.append(
            (
                "context_distraction",
                perturbed,
            )
        )

    return perturbations


def generate_sentiment_insertion(
    sentence: str,
    aspect: str,
) -> list[tuple[str, str]]:

    perturbations = []

    # Insert sentiment modifiers immediately before
    # the aspect. This creates a targeted local-context
    # perturbation.
    for word in INTENSIFIERS:

        perturbed = safe_insert_before(
            sentence,
            aspect,
            word,
        )

        if perturbed:

            perturbations.append(
                (
                    "intensifier_near_aspect",
                    perturbed,
                )
            )

    for word in DIMINISHERS:

        perturbed = safe_insert_before(
            sentence,
            aspect,
            word,
        )

        if perturbed:

            perturbations.append(
                (
                    "diminisher_near_aspect",
                    perturbed,
                )
            )

    return perturbations


def generate_negation_perturbations(
    sentence: str,
    aspect: str,
) -> list[tuple[str, str]]:

    perturbations = []

    for phrase in NEGATION_PHRASES:

        perturbed = safe_insert_before(
            sentence,
            aspect,
            phrase,
        )

        if perturbed:

            perturbations.append(
                (
                    "negation_near_aspect",
                    perturbed,
                )
            )

    return perturbations


def generate_aspect_context_interference(
    sentence: str,
    aspect: str,
) -> list[tuple[str, str]]:

    perturbations = []

    # Insert an additional aspect-like contextual phrase
    # immediately around the target aspect.
    interference_phrases = [
        "the other",
        "this particular",
        "the mentioned",
    ]

    for phrase in interference_phrases:

        perturbed = safe_insert_before(
            sentence,
            aspect,
            phrase,
        )

        if perturbed:

            perturbations.append(
                (
                    "aspect_context_interference",
                    perturbed,
                )
            )

    return perturbations


def generate_sentiment_word_interference(
    sentence: str,
    aspect: str,
) -> list[tuple[str, str]]:

    perturbations = []

    sentiment_words = find_sentiment_words(
        sentence
    )

    # Add a contrasting sentiment cue at the end.
    #
    # This does NOT change the gold label. The purpose is
    # to test whether the model is distracted by an
    # additional sentiment-bearing expression.
    if sentiment_words:

        for phrase in [
            "although another part was great",
            "although another part was disappointing",
            "despite some positive comments",
            "despite some negative comments",
        ]:

            perturbed = (
                sentence.rstrip()
                + " "
                + phrase
                + "."
            )

            perturbations.append(
                (
                    "sentiment_word_interference",
                    perturbed,
                )
            )

    return perturbations


def generate_local_context_shift(
    sentence: str,
    aspect: str,
) -> list[tuple[str, str]]:

    perturbations = []

    insertions = [
        "in particular",
        "for this item",
        "from this perspective",
    ]

    for insertion in insertions:

        perturbed = safe_insert_after(
            sentence,
            aspect,
            insertion,
        )

        if perturbed:

            perturbations.append(
                (
                    "local_context_shift",
                    perturbed,
                )
            )

    return perturbations


# ============================================================
# ALL PERTURBATIONS FOR ONE INSTANCE
# ============================================================

def generate_perturbations(
    sentence: str,
    aspect: str,
) -> list[tuple[str, str]]:

    perturbations = []

    perturbations.extend(
        generate_context_distraction(
            sentence,
            aspect,
        )
    )

    perturbations.extend(
        generate_sentiment_insertion(
            sentence,
            aspect,
        )
    )

    perturbations.extend(
        generate_negation_perturbations(
            sentence,
            aspect,
        )
    )

    perturbations.extend(
        generate_aspect_context_interference(
            sentence,
            aspect,
        )
    )

    perturbations.extend(
        generate_sentiment_word_interference(
            sentence,
            aspect,
        )
    )

    perturbations.extend(
        generate_local_context_shift(
            sentence,
            aspect,
        )
    )

    return perturbations


# ============================================================
# MAIN GENERATION
# ============================================================

def main():

    print("=" * 70)
    print("ROBUSTNESS EXPERIMENT 3")
    print("TARGETED ADVERSARIAL PERTURBATION GENERATION")
    print("=" * 70)

    print()
    print("Input file:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()
    print("Loading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

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
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    print(
        f"Original rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Generate perturbations
    # --------------------------------------------------------

    print()
    print("Generating targeted adversarial perturbations...")

    records = []

    perturbation_counter = 1

    valid_originals = 0

    for index, row in df.iterrows():

        sentence_id = str(
            row["sentence_id"]
        )

        sentence = str(
            row["sentence"]
        )

        aspect = str(
            row["aspect"]
        )

        polarity = str(
            row["polarity"]
        )

        label = int(
            row["label"]
        )

        generated = generate_perturbations(
            sentence,
            aspect,
        )

        instance_count = 0

        seen_texts = set()

        for perturbation_type, perturbed_sentence in generated:

            perturbed_sentence = (
                normalize_text(
                    perturbed_sentence
                )
            )

            if perturbed_sentence in seen_texts:
                continue

            seen_texts.add(
                perturbed_sentence
            )

            if not validate_perturbation(
                sentence,
                perturbed_sentence,
                aspect,
            ):
                continue

            records.append(
                {
                    "perturbation_id": (
                        f"ADV_{perturbation_counter:07d}"
                    ),
                    "sentence_id": sentence_id,
                    "aspect": aspect,
                    "true_polarity": polarity,
                    "true_label": label,
                    "perturbation_type": (
                        perturbation_type
                    ),
                    "original_sentence": sentence,
                    "perturbed_sentence": (
                        perturbed_sentence
                    ),
                    "aspect_preserved": True,
                }
            )

            perturbation_counter += 1
            instance_count += 1

        if instance_count > 0:
            valid_originals += 1

        if (
            (index + 1) % 500 == 0
            or index + 1 == len(df)
        ):

            print(
                f"Processed {index + 1:,}/"
                f"{len(df):,} instances | "
                f"Generated: {len(records):,}"
            )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        records
    )

    if result_df.empty:

        raise RuntimeError(
            "No targeted perturbations were generated."
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TARGETED PERTURBATION VALIDATION")
    print("=" * 70)

    aspect_preservation_rate = (
        result_df["aspect_preserved"]
        .mean()
    )

    unique_originals = (
        result_df["sentence_id"]
        .nunique()
    )

    print(
        f"Original instances       : {len(df):,}"
    )

    print(
        f"Eligible original instances: "
        f"{valid_originals:,}"
    )

    print(
        f"Perturbed instances       : "
        f"{len(result_df):,}"
    )

    print(
        f"Unique original instances : "
        f"{unique_originals:,}"
    )

    print(
        f"Aspect preservation rate  : "
        f"{aspect_preservation_rate:.4f}"
    )

    print()
    print("Duplicate perturbations removed.")

    # --------------------------------------------------------
    # Perturbation summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TARGETED PERTURBATION SUMMARY")
    print("=" * 70)

    perturbation_counts = (
        result_df[
            "perturbation_type"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        perturbation_counts.to_string()
    )

    # --------------------------------------------------------
    # Polarity distribution
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("POLARITY DISTRIBUTION")
    print("=" * 70)

    polarity_counts = (
        result_df[
            "true_polarity"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        polarity_counts.to_string()
    )

    # --------------------------------------------------------
    # Aspect preservation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ASPECT PRESERVATION")
    print("=" * 70)

    preservation_counts = (
        result_df[
            "aspect_preserved"
        ]
        .value_counts()
    )

    print(
        preservation_counts.to_string()
    )

    # --------------------------------------------------------
    # Average perturbations
    # --------------------------------------------------------

    average_per_instance = (
        len(result_df)
        / unique_originals
        if unique_originals > 0
        else 0.0
    )

    print()
    print(
        f"Average perturbations / "
        f"original instance : "
        f"{average_per_instance:.2f}"
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {
        "experiment": "Robustness Experiment 3",
        "name": (
            "Targeted Adversarial "
            "Perturbation Generation"
        ),
        "seed": SEED,
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "original_instances": int(len(df)),
        "eligible_original_instances": int(
            valid_originals
        ),
        "perturbed_instances": int(
            len(result_df)
        ),
        "unique_original_instances": int(
            unique_originals
        ),
        "average_perturbations_per_instance": float(
            average_per_instance
        ),
        "aspect_preservation_rate": float(
            aspect_preservation_rate
        ),
        "perturbation_distribution": {
            str(k): int(v)
            for k, v in perturbation_counts.items()
        },
        "polarity_distribution": {
            str(k): int(v)
            for k, v in polarity_counts.items()
        },
        "aspect_preservation": {
            str(k): int(v)
            for k, v in preservation_counts.items()
        },
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print("Saved:")
    print(SUMMARY_FILE)

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE TARGETED PERTURBATIONS")
    print("=" * 70)

    sample_columns = [
        "perturbation_id",
        "sentence_id",
        "aspect",
        "true_polarity",
        "perturbation_type",
        "original_sentence",
        "perturbed_sentence",
    ]

    print(
        result_df[
            sample_columns
        ]
        .head(20)
        .to_string(
            index=False,
            max_colwidth=100,
        )
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TARGETED ADVERSARIAL "
        "PERTURBATION GENERATION COMPLETE"
    )
    print("=" * 70)

    print()
    print("Generated files:")
    print(
        " - targeted_adversarial_instances.csv"
    )
    print(
        " - targeted_adversarial_summary.json"
    )

    print()
    print("Next step:")
    print(
        "Run the original and targeted adversarial "
        "sentences through the DistilBERT baseline "
        "and measure prediction consistency, flip "
        "rate, confidence change, and performance "
        "degradation."
    )


if __name__ == "__main__":
    main()