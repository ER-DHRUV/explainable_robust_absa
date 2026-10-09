 """
controlled_perturbation_generation.py

ROBUSTNESS EXPERIMENT 1
Controlled Perturbation Generation

Purpose
-------
Generate controlled, primarily label-preserving perturbations for
ABSA robustness evaluation.

The generated data will later be passed through the same DistilBERT
baseline used on the original test set.

Perturbation families
---------------------
1. Context insertion
2. Context removal
3. Neutral context insertion
4. Intensifier insertion
5. Diminisher insertion
6. Irrelevant token noise
7. Punctuation variation
8. Whitespace variation

Important
---------
The aspect term is preserved whenever possible.

This script DOES NOT run model inference.
It only creates the controlled robustness dataset.

Output
------
results/transformer/distilbert_baseline/robustness/
    controlled_perturbations/
        perturbed_instances.csv
        perturbation_summary.json
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "results/transformer/distilbert_baseline"
)

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------
#
# Change this path only if your original MAMS test file has a
# different location.
#
# The script accepts a CSV containing at least:
#
# sentence_id
# aspect
# sentence
# true_polarity
#
# ------------------------------------------------------------

INPUT_FILE = Path(
    "C:/Users/dhruv/Desktop/explainable_robust_absa/data/processed/test.csv"
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    BASE_DIR
    / "robustness"
    / "controlled_perturbations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "perturbed_instances.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "perturbation_summary.json"
)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# CONFIGURATION OF PERTURBATIONS
# ============================================================

# Number of generated perturbations per original instance.
#
# Set to None to generate every enabled perturbation.
MAX_PERTURBATIONS_PER_INSTANCE = None


# ------------------------------------------------------------
# Label-preserving context
# ------------------------------------------------------------

NEUTRAL_CONTEXTS = [
    "In the review,",
    "According to the customer,",
    "Overall,",
    "In this case,",
    "For this visit,",
    "From the review,",
]


# ------------------------------------------------------------
# Neutral sentence additions
# ------------------------------------------------------------

NEUTRAL_SENTENCE_SUFFIXES = [
    "The visit was on a weekday.",
    "The customer mentioned this in the review.",
    "This was part of the overall review.",
    "The comment was included in the review.",
]


# ------------------------------------------------------------
# Mild intensifier/diminisher markers
#
# These are deliberately kept outside the sentiment-bearing
# aspect phrase whenever possible.
#
# NOTE:
# These transformations are marked as label-preserving
# candidates, not guaranteed semantic invariants.
# ------------------------------------------------------------

INTENSIFIERS = [
    "really",
    "quite",
    "especially",
]

DIMINISHERS = [
    "somewhat",
    "relatively",
    "rather",
]


# ------------------------------------------------------------
# Irrelevant filler tokens
# ------------------------------------------------------------

NOISE_TOKENS = [
    "indeed",
    "overall",
    "notably",
    "personally",
]


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "sentence_id",
    "aspect",
    "sentence",
    "true_polarity",
]


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_polarity(value):
    """
    Normalize polarity labels.
    """

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    return value


def clean_text(value):
    """
    Convert a value into a clean string.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_whitespace(text):
    """
    Normalize repeated whitespace.
    """

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# ============================================================
# ASPECT MATCHING
# ============================================================

def find_aspect_span(sentence, aspect):
    """
    Locate the aspect in the sentence.

    Matching is case-insensitive.

    Returns:
        (start, end)

    If the aspect cannot be located:
        (None, None)
    """

    sentence = clean_text(sentence)
    aspect = clean_text(aspect)

    if not sentence or not aspect:
        return None, None

    pattern = re.escape(aspect)

    match = re.search(
        pattern,
        sentence,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None, None

    return (
        match.start(),
        match.end(),
    )


def aspect_exists(sentence, aspect):
    """
    Check whether the aspect is present in the sentence.
    """

    start, end = find_aspect_span(
        sentence,
        aspect,
    )

    return (
        start is not None
        and end is not None
    )


# ============================================================
# PERTURBATION 1
# ============================================================

def perturb_context_insertion(sentence, aspect):
    """
    Insert neutral context before the sentence.

    Example:

        Original:
        "The food was excellent."

        Perturbed:
        "According to the customer, the food was excellent."
    """

    if not aspect_exists(sentence, aspect):
        return None

    context = random.choice(
        NEUTRAL_CONTEXTS
    )

    perturbed = (
        f"{context} "
        f"{sentence}"
    )

    return normalize_whitespace(
        perturbed
    )


# ============================================================
# PERTURBATION 2
# ============================================================

def perturb_neutral_suffix(sentence, aspect):
    """
    Add a neutral sentence after the original sentence.

    Example:

        "The food was excellent."
        ->
        "The food was excellent. The customer
        mentioned this in the review."
    """

    if not aspect_exists(sentence, aspect):
        return None

    suffix = random.choice(
        NEUTRAL_SENTENCE_SUFFIXES
    )

    sentence = sentence.rstrip()

    if sentence.endswith(
        (".", "!", "?")
    ):
        perturbed = (
            f"{sentence} {suffix}"
        )
    else:
        perturbed = (
            f"{sentence}. {suffix}"
        )

    return normalize_whitespace(
        perturbed
    )


# ============================================================
# PERTURBATION 3
# ============================================================

def perturb_intensifier(sentence, aspect):
    """
    Insert a mild intensifier immediately before the
    first adjective/adverb-like sentiment word when a
    simple candidate can be identified.

    This is deliberately conservative.

    If no suitable candidate is found, return None.
    """

    if not aspect_exists(sentence, aspect):
        return None

    # Common sentiment-bearing words found in restaurant
    # review-style MAMS data.
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
        return None

    intensifier = random.choice(
        INTENSIFIERS
    )

    original_word = match.group(0)

    replacement = (
        f"{intensifier} "
        f"{original_word}"
    )

    perturbed = (
        sentence[:match.start()]
        + replacement
        + sentence[match.end():]
    )

    return normalize_whitespace(
        perturbed
    )


# ============================================================
# PERTURBATION 4
# ============================================================

def perturb_diminisher(sentence, aspect):
    """
    Insert a mild diminisher before a sentiment word.

    This is treated as a controlled lexical perturbation.
    """

    if not aspect_exists(sentence, aspect):
        return None

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
        return None

    diminisher = random.choice(
        DIMINISHERS
    )

    original_word = match.group(0)

    replacement = (
        f"{diminisher} "
        f"{original_word}"
    )

    perturbed = (
        sentence[:match.start()]
        + replacement
        + sentence[match.end():]
    )

    return normalize_whitespace(
        perturbed
    )


# ============================================================
# PERTURBATION 5
# ============================================================

def perturb_irrelevant_noise(sentence, aspect):
    """
    Insert a semantically irrelevant discourse token
    while preserving the original sentence content.
    """

    if not aspect_exists(sentence, aspect):
        return None

    noise = random.choice(
        NOISE_TOKENS
    )

    sentence = sentence.strip()

    if sentence.endswith(
        (".", "!", "?")
    ):
        punctuation = sentence[-1]
        body = sentence[:-1].rstrip()

        perturbed = (
            f"{body}, {noise}{punctuation}"
        )

    else:

        perturbed = (
            f"{sentence}, {noise}"
        )

    return normalize_whitespace(
        perturbed
    )


# ============================================================
# PERTURBATION 6
# ============================================================

def perturb_punctuation(sentence, aspect):
    """
    Controlled punctuation variation.

    Examples:

        "The food was good."
        ->
        "The food was good!"

        "The food was good!"
        ->
        "The food was good."
    """

    if not aspect_exists(sentence, aspect):
        return None

    sentence = sentence.strip()

    if sentence.endswith("."):

        perturbed = (
            sentence[:-1] + "!"
        )

    elif sentence.endswith("!"):

        perturbed = (
            sentence[:-1] + "."
        )

    elif sentence.endswith("?"):

        perturbed = (
            sentence[:-1] + "."
        )

    else:

        perturbed = (
            sentence + "."
        )

    return perturbed


# ============================================================
# PERTURBATION 7
# ============================================================

def perturb_whitespace(sentence, aspect):
    """
    Controlled whitespace variation.

    Multiple spaces are inserted between selected words.

    This should not change the underlying lexical content.
    """

    if not aspect_exists(sentence, aspect):
        return None

    words = sentence.split()

    if len(words) < 3:
        return None

    # Select a position away from the beginning/end.
    position = random.randint(
        1,
        len(words) - 2,
    )

    words[position] = (
        "  " + words[position]
    )

    perturbed = " ".join(words)

    return perturbed


# ============================================================
# PERTURBATION REGISTRY
# ============================================================

PERTURATIONS = [
    {
        "name": "context_insertion",
        "function": perturb_context_insertion,
        "label_preserving": True,
        "description": (
            "Insert neutral discourse context before "
            "the original sentence."
        ),
    },
    {
        "name": "neutral_suffix",
        "function": perturb_neutral_suffix,
        "label_preserving": True,
        "description": (
            "Append a neutral sentence that does not "
            "modify the original sentiment."
        ),
    },
    {
        "name": "intensifier_insertion",
        "function": perturb_intensifier,
        "label_preserving": True,
        "description": (
            "Insert a mild intensifier before a "
            "recognized sentiment-bearing word."
        ),
    },
    {
        "name": "diminisher_insertion",
        "function": perturb_diminisher,
        "label_preserving": True,
        "description": (
            "Insert a mild diminisher before a "
            "recognized sentiment-bearing word."
        ),
    },
    {
        "name": "irrelevant_noise",
        "function": perturb_irrelevant_noise,
        "label_preserving": True,
        "description": (
            "Add a semantically irrelevant discourse "
            "token."
        ),
    },
    {
        "name": "punctuation_variation",
        "function": perturb_punctuation,
        "label_preserving": True,
        "description": (
            "Change sentence-final punctuation without "
            "changing lexical content."
        ),
    },
    {
        "name": "whitespace_variation",
        "function": perturb_whitespace,
        "label_preserving": True,
        "description": (
            "Introduce harmless whitespace variation."
        ),
    },
]


# ============================================================
# GENERATE PERTURBATIONS FOR ONE INSTANCE
# ============================================================

def generate_instance_perturbations(
    row,
    max_perturbations=None,
):
    """
    Generate controlled perturbations for one ABSA
    instance.
    """

    sentence_id = row["sentence_id"]
    aspect = row["aspect"]
    sentence = row["sentence"]
    true_polarity = row["true_polarity"]

    results = []

    selected_perturbations = list(
        PERTURATIONS

    )

    if max_perturbations is not None:

        if max_perturbations < len(
            selected_perturbations
        ):

            selected_perturbations = random.sample(
                selected_perturbations,
                max_perturbations,
            )

    for perturbation in selected_perturbations:

        function = perturbation[
            "function"
        ]

        try:

            perturbed_sentence = function(
                sentence,
                aspect,
            )

        except Exception as error:

            print(
                f"Warning: perturbation "
                f"{perturbation['name']} failed "
                f"for sentence {sentence_id}: "
                f"{error}"
            )

            continue

        if perturbed_sentence is None:
            continue

        perturbed_sentence = clean_text(
            perturbed_sentence
        )

        if not perturbed_sentence:
            continue

        if perturbed_sentence == sentence:
            continue

        # ----------------------------------------------------
        # Verify aspect preservation
        # ----------------------------------------------------

        aspect_preserved = aspect_exists(
            perturbed_sentence,
            aspect,
        )

        # ----------------------------------------------------
        # Create result
        # ----------------------------------------------------

        results.append(
            {
                "sentence_id": sentence_id,
                "aspect": aspect,
                "true_polarity": true_polarity,

                "original_sentence": sentence,

                "perturbed_sentence": (
                    perturbed_sentence
                ),

                "perturbation_type": (
                    perturbation["name"]
                ),

                "perturbation_description": (
                    perturbation["description"]
                ),

                "label_preserving_expected": (
                    perturbation[
                        "label_preserving"
                    ]
                ),

                "aspect_preserved": (
                    aspect_preserved
                ),

                "perturbation_strength": "mild",

                "random_seed": SEED,
            }
        )

    return results


# ============================================================
# VALIDATE INPUT
# ============================================================

print("=" * 70)
print("ROBUSTNESS EXPERIMENT 1")
print("CONTROLLED PERTURBATION GENERATION")
print("=" * 70)

print("\nInput file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nInput file not found:\n"
        f"{INPUT_FILE}\n\n"
        "Update INPUT_FILE in the configuration "
        "section to point to your MAMS test CSV."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE
)
# Dataset uses "polarity" as the ground-truth sentiment label.
# Normalize it to the name expected by the robustness pipeline.
if "true_polarity" not in df.columns and "polarity" in df.columns:
    df["true_polarity"] = df["polarity"]

print(
    f"Original rows: {len(df):,}"
)

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# NORMALIZE INPUT
# ============================================================

for column in [
    "sentence_id",
    "aspect",
    "sentence",
    "true_polarity",
]:

    df[column] = df[column].apply(
        clean_text
    )

df["true_polarity"] = (
    df["true_polarity"]
    .apply(normalize_polarity)
)


# ============================================================
# REMOVE INVALID INSTANCES
# ============================================================

initial_count = len(df)

df = df[
    (
        df["sentence"] != ""
    )
    &
    (
        df["aspect"] != ""
    )
].copy()

removed_count = (
    initial_count - len(df)
)

if removed_count > 0:

    print(
        f"Removed invalid rows: "
        f"{removed_count:,}"
    )


# ============================================================
# GENERATE
# ============================================================

print("\nGenerating controlled perturbations...")

all_results = []

total_instances = len(df)

for counter, (_, row) in enumerate(
    df.iterrows(),
    start=1,
):

    results = generate_instance_perturbations(
        row,
        max_perturbations=(
            MAX_PERTURBATIONS_PER_INSTANCE
        ),
    )

    all_results.extend(
        results
    )

    if (
        counter % 500 == 0
        or counter == total_instances
    ):

        print(
            f"Processed "
            f"{counter:,}/"
            f"{total_instances:,} "
            f"instances | "
            f"Generated: "
            f"{len(all_results):,}"
        )


# ============================================================
# BUILD OUTPUT DATAFRAME
# ============================================================

perturbed_df = pd.DataFrame(
    all_results
)


# ============================================================
# ADD PERTURBATION INSTANCE ID
# ============================================================

if len(perturbed_df) > 0:

    perturbed_df.insert(
        0,
        "perturbation_id",
        [
            f"ROBUST_{index:07d}"
            for index in range(
                1,
                len(perturbed_df) + 1,
            )
        ],
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

if len(perturbed_df) > 0:

    aspect_preservation_rate = (
        perturbed_df[
            "aspect_preserved"
        ].mean()
    )

else:

    aspect_preservation_rate = 0.0


print("\n" + "=" * 70)
print("PERTURBATION VALIDATION")
print("=" * 70)

print(
    f"Original instances       : "
    f"{len(df):,}"
)

print(
    f"Perturbed instances      : "
    f"{len(perturbed_df):,}"
)

print(
    f"Aspect preservation rate : "
    f"{aspect_preservation_rate:.4f}"
)


# ============================================================
# SAVE CSV
# ============================================================

if len(perturbed_df) == 0:

    raise RuntimeError(
        "No perturbations were generated. "
        "Check the input dataset and aspect matching."
    )


perturbed_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8",
)

print(
    f"\nSaved:\n{OUTPUT_FILE}"
)


# ============================================================
# PERTURBATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PERTURBATION SUMMARY")
print("=" * 70)

type_counts = (
    perturbed_df[
        "perturbation_type"
    ]
    .value_counts()
    .sort_index()
)

print(
    type_counts.to_string()
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("POLARITY DISTRIBUTION")
print("=" * 70)

polarity_counts = (
    perturbed_df[
        "true_polarity"
    ]
    .value_counts()
    .sort_index()
)

print(
    polarity_counts.to_string()
)


# ============================================================
# ASPECT PRESERVATION
# ============================================================

print("\n" + "=" * 70)
print("ASPECT PRESERVATION")
print("=" * 70)

aspect_counts = (
    perturbed_df[
        "aspect_preserved"
    ]
    .value_counts()
)

print(
    aspect_counts.to_string()
)


# ============================================================
# INSTANCES WITH PERTURBATIONS
# ============================================================

unique_original_instances = (
    perturbed_df[
        "sentence_id"
    ]
    .nunique()
)

average_perturbations = (
    len(perturbed_df)
    / unique_original_instances
    if unique_original_instances > 0
    else 0
)

print(
    f"\nUnique original instances : "
    f"{unique_original_instances:,}"
)

print(
    f"Average perturbations / "
    f"instance                 : "
    f"{average_perturbations:.2f}"
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "experiment": (
        "Robustness Experiment 1 - "
        "Controlled Perturbation Generation"
    ),

    "seed": SEED,

    "input_file": str(
        INPUT_FILE
    ),

    "output_file": str(
        OUTPUT_FILE
    ),

    "original_instances": int(
        len(df)
    ),

    "generated_perturbations": int(
        len(perturbed_df)
    ),

    "unique_original_instances": int(
        unique_original_instances
    ),

    "average_perturbations_per_instance": (
        float(average_perturbations)
    ),

    "aspect_preservation_rate": (
        float(aspect_preservation_rate)
    ),

    "label_preserving_expected": True,

    "perturbation_types": {
        str(key): int(value)
        for key, value
        in type_counts.items()
    },

    "polarity_distribution": {
        str(key): int(value)
        for key, value
        in polarity_counts.items()
    },

    "perturbation_definitions": {
        perturbation["name"]: {
            "label_preserving_expected": (
                perturbation[
                    "label_preserving"
                ]
            ),
            "description": (
                perturbation[
                    "description"
                ]
            ),
        }
        for perturbation
        in PERTURATIONS

    },

    "notes": [
        (
            "This experiment generates controlled "
            "perturbations and does not perform "
            "model inference."
        ),
        (
            "The aspect term is preserved whenever "
            "the perturbation can be generated."
        ),
        (
            "Label-preserving status indicates the "
            "intended experimental design; semantic "
            "invariance should be manually validated "
            "for sampled examples."
        ),
        (
            "The resulting confidence and prediction "
            "changes will be measured in the next "
            "robustness experiment."
        ),
    ],
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=4,
    )


print(
    f"\nSaved:\n{SUMMARY_FILE}"
)


# ============================================================
# SAMPLE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE GENERATED PERTURBATIONS")
print("=" * 70)

sample_columns = [
    "perturbation_id",
    "aspect",
    "true_polarity",
    "perturbation_type",
    "original_sentence",
    "perturbed_sentence",
]

print(
    perturbed_df[
        sample_columns
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CONTROLLED PERTURBATION GENERATION COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    f" - {OUTPUT_FILE.name}"
)

print(
    f" - {SUMMARY_FILE.name}"
)

print(
    "\nNext step:"
)

print(
    "Run the original and perturbed sentences "
    "through the DistilBERT baseline and measure "
    "prediction consistency, flip rate, and "
    "performance degradation."
)