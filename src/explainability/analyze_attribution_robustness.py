import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "results/transformer/distilbert_baseline/"
    "explainability/explanation_results.csv"
)

OUTPUT_DIR = Path(
    "results/transformer/distilbert_baseline/"
    "explainability/attribution_robustness"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def clean_token(token):
    """Normalize token text for matching and readability."""
    if pd.isna(token):
        return ""

    token = str(token).strip()

    # Remove common WordPiece / BPE markers
    token = token.replace("##", "")
    token = token.replace("Ġ", "")
    token = token.replace("▁", "")

    return token.lower()


def tokenize_aspect(aspect):
    """Return normalized aspect words."""
    if pd.isna(aspect):
        return set()

    words = re.findall(
        r"[a-zA-Z0-9]+",
        str(aspect).lower()
    )

    return set(words)


def safe_mean(series):
    if len(series) == 0:
        return 0.0

    return float(series.mean())


def safe_median(series):
    if len(series) == 0:
        return 0.0

    return float(series.median())


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SHAP ANALYSIS — EXPERIMENT 5")
print("ATTRIBUTION ROBUSTNESS")
print("=" * 70)

print("\nLoading SHAP explanations...")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

required_columns = {
    "sentence_id",
    "aspect",
    "true_polarity",
    "predicted_polarity",
    "token",
    "shap_predicted_class",
    "correct",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(sorted(missing))
    )

print(f"Rows      : {len(df):,}")

instance_count = (
    df[["sentence_id", "aspect"]]
    .drop_duplicates()
    .shape[0]
)

print(f"Instances : {instance_count:,}")


# ============================================================
# PREPARE TOKENS
# ============================================================

df["token_clean"] = df["token"].apply(clean_token)

df["shap_abs"] = (
    df["shap_predicted_class"].abs()
)

df["aspect_words"] = (
    df["aspect"].apply(tokenize_aspect)
)

df["is_aspect_token"] = df.apply(
    lambda row: (
        row["token_clean"] in row["aspect_words"]
        and row["token_clean"] != ""
    ),
    axis=1,
)

df["is_context_token"] = ~df["is_aspect_token"]


# ============================================================
# INSTANCE-LEVEL ATTRIBUTION PROFILE
# ============================================================

print("\n" + "=" * 70)
print("BUILDING ATTRIBUTION PROFILES")
print("=" * 70)

records = []

group_columns = [
    "sentence_id",
    "aspect",
]

grouped = df.groupby(
    group_columns,
    sort=False
)

for i, ((sentence_id, aspect), g) in enumerate(
    grouped,
    start=1
):

    if i % 100 == 0:
        print(
            f"Processed {i:,}/{instance_count:,} instances"
        )

    correct = bool(
        g["correct"].iloc[0]
    )

    true_polarity = (
        g["true_polarity"].iloc[0]
    )

    predicted_polarity = (
        g["predicted_polarity"].iloc[0]
    )

    total_abs = g["shap_abs"].sum()

    aspect_abs = g.loc[
        g["is_aspect_token"],
        "shap_abs"
    ].sum()

    context_abs = g.loc[
        g["is_context_token"],
        "shap_abs"
    ].sum()

    if total_abs > 0:
        aspect_ratio = (
            aspect_abs / total_abs
        )

        context_ratio = (
            context_abs / total_abs
        )

    else:
        aspect_ratio = 0.0
        context_ratio = 0.0

    # --------------------------------------------------------
    # Attribution concentration
    # --------------------------------------------------------

    sorted_abs = np.sort(
        g["shap_abs"].to_numpy()
    )[::-1]

    if (
        len(sorted_abs) > 0
        and sorted_abs.sum() > 0
    ):
        top1_share = (
            sorted_abs[0]
            / sorted_abs.sum()
        )

    else:
        top1_share = 0.0

    if (
        len(sorted_abs) >= 3
        and sorted_abs.sum() > 0
    ):
        top3_share = (
            sorted_abs[:3].sum()
            / sorted_abs.sum()
        )

    else:
        top3_share = (
            sorted_abs.sum()
            / sorted_abs.sum()
            if (
                len(sorted_abs) > 0
                and sorted_abs.sum() > 0
            )
            else 0.0
        )

    # --------------------------------------------------------
    # Effective attribution spread
    # --------------------------------------------------------

    proportions = (
        g["shap_abs"] / total_abs
        if total_abs > 0
        else pd.Series(dtype=float)
    )

    if len(proportions) > 0:
        entropy = float(
            -(
                proportions
                * np.log2(proportions + 1e-12)
            ).sum()
        )

    else:
        entropy = 0.0

    # Normalize entropy by number of tokens
    if len(proportions) > 1:
        normalized_entropy = (
            entropy
            / np.log2(len(proportions))
        )

    else:
        normalized_entropy = 0.0

    # --------------------------------------------------------
    # Dominance / gap
    # --------------------------------------------------------

    attribution_gap = (
        context_ratio - aspect_ratio
    )

    context_dominant = (
        context_ratio > aspect_ratio
    )

    # --------------------------------------------------------
    # Strongest token
    # --------------------------------------------------------

    top_row = g.loc[
        g["shap_abs"].idxmax()
    ]

    strongest_token = (
        top_row["token_clean"]
    )

    strongest_shap = float(
        top_row["shap_predicted_class"]
    )

    # --------------------------------------------------------
    # Aspect/context token counts
    # --------------------------------------------------------

    aspect_token_count = int(
        g["is_aspect_token"].sum()
    )

    context_token_count = int(
        g["is_context_token"].sum()
    )

    records.append(
        {
            "sentence_id": sentence_id,
            "aspect": aspect,
            "true_polarity": true_polarity,
            "predicted_polarity": predicted_polarity,
            "correct": correct,
            "total_tokens": len(g),
            "aspect_token_count": aspect_token_count,
            "context_token_count": context_token_count,
            "aspect_abs_shap": float(aspect_abs),
            "context_abs_shap": float(context_abs),
            "aspect_attribution_ratio": float(
                aspect_ratio
            ),
            "context_attribution_ratio": float(
                context_ratio
            ),
            "attribution_gap": float(
                attribution_gap
            ),
            "context_dominant": bool(
                context_dominant
            ),
            "top1_attribution_share": float(
                top1_share
            ),
            "top3_attribution_share": float(
                top3_share
            ),
            "attribution_entropy": float(
                entropy
            ),
            "normalized_attribution_entropy": float(
                normalized_entropy
            ),
            "strongest_token": strongest_token,
            "strongest_shap": strongest_shap,
        }
    )


profiles = pd.DataFrame(records)

profile_path = (
    OUTPUT_DIR
    / "attribution_robustness_by_instance.csv"
)

profiles.to_csv(
    profile_path,
    index=False
)

print(
    f"\nSaved: {profile_path}"
)


# ============================================================
# ROBUSTNESS SCORE
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING ROBUSTNESS PROXY")
print("=" * 70)

"""
Interpretation:

A robust explanation should ideally avoid being dominated
by a single context token.

We therefore construct a diagnostic robustness score:

    robustness =
        1
        - top1 attribution share
        - context dominance penalty

The score is NOT a model-quality metric.

It is only a diagnostic indicator of attribution
concentration/context dependence.

Higher values indicate less concentrated/context-dominated
attribution.
"""

profiles["context_penalty"] = (
    profiles["context_attribution_ratio"]
)

profiles["concentration_penalty"] = (
    profiles["top1_attribution_share"]
)

profiles["robustness_proxy"] = (
    1.0
    - 0.5 * profiles["context_penalty"]
    - 0.5 * profiles["concentration_penalty"]
)

profiles["robustness_proxy"] = (
    profiles["robustness_proxy"]
    .clip(0.0, 1.0)
)

profiles.to_csv(
    profile_path,
    index=False
)


# ============================================================
# CORRECT VS INCORRECT
# ============================================================

print("\n" + "=" * 70)
print("CORRECT VS INCORRECT")
print("=" * 70)

summary = (
    profiles
    .groupby("correct")
    .agg(
        instances=(
            "sentence_id",
            "count"
        ),
        mean_robustness_proxy=(
            "robustness_proxy",
            "mean"
        ),
        median_robustness_proxy=(
            "robustness_proxy",
            "median"
        ),
        mean_top1_attribution_share=(
            "top1_attribution_share",
            "mean"
        ),
        median_top1_attribution_share=(
            "top1_attribution_share",
            "median"
        ),
        mean_top3_attribution_share=(
            "top3_attribution_share",
            "mean"
        ),
        mean_normalized_entropy=(
            "normalized_attribution_entropy",
            "mean"
        ),
        mean_context_attribution_ratio=(
            "context_attribution_ratio",
            "mean"
        ),
        mean_aspect_attribution_ratio=(
            "aspect_attribution_ratio",
            "mean"
        ),
        mean_attribution_gap=(
            "attribution_gap",
            "mean"
        ),
        context_dominant_instances=(
            "context_dominant",
            "sum"
        ),
    )
    .reset_index()
)

summary["context_dominant_rate"] = (
    summary["context_dominant_instances"]
    / summary["instances"]
)

print(
    summary.to_string(index=False)
)

summary_path = (
    OUTPUT_DIR
    / "attribution_robustness_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    f"\nSaved: {summary_path}"
)


# ============================================================
# ERROR TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ERROR TYPE ANALYSIS")
print("=" * 70)

errors = profiles[
    profiles["correct"] == False
].copy()

error_type = (
    errors
    .groupby(
        [
            "true_polarity",
            "predicted_polarity",
        ]
    )
    .agg(
        instances=(
            "sentence_id",
            "count"
        ),
        mean_robustness_proxy=(
            "robustness_proxy",
            "mean"
        ),
        mean_top1_attribution_share=(
            "top1_attribution_share",
            "mean"
        ),
        mean_top3_attribution_share=(
            "top3_attribution_share",
            "mean"
        ),
        mean_normalized_entropy=(
            "normalized_attribution_entropy",
            "mean"
        ),
        mean_context_attribution_ratio=(
            "context_attribution_ratio",
            "mean"
        ),
        mean_aspect_attribution_ratio=(
            "aspect_attribution_ratio",
            "mean"
        ),
        mean_attribution_gap=(
            "attribution_gap",
            "mean"
        ),
        context_dominant_instances=(
            "context_dominant",
            "sum"
        ),
    )
    .reset_index()
)

error_type["context_dominant_rate"] = (
    error_type["context_dominant_instances"]
    / error_type["instances"]
)

print(
    error_type.to_string(index=False)
)

error_type_path = (
    OUTPUT_DIR
    / "attribution_robustness_by_error_type.csv"
)

error_type.to_csv(
    error_type_path,
    index=False
)

print(
    f"\nSaved: {error_type_path}"
)


# ============================================================
# UNSTABLE ATTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFYING UNSTABLE ATTRIBUTIONS")
print("=" * 70)

"""
An attribution is flagged as unstable when:

1. The strongest token explains a large fraction
   of the total attribution.

2. Context attribution strongly dominates
   aspect attribution.

3. The instance is misclassified.

These are diagnostic conditions, not proof
of causal instability.
"""

unstable = profiles[
    (
        (profiles["correct"] == False)
        & (
            profiles["top1_attribution_share"]
            >= 0.25
        )
        & (
            profiles["context_attribution_ratio"]
            >= 0.80
        )
    )
].copy()

unstable = unstable.sort_values(
    [
        "robustness_proxy",
        "context_attribution_ratio",
        "top1_attribution_share",
    ],
    ascending=[
        True,
        False,
        False,
    ],
)

unstable_columns = [
    "sentence_id",
    "aspect",
    "true_polarity",
    "predicted_polarity",
    "robustness_proxy",
    "context_attribution_ratio",
    "aspect_attribution_ratio",
    "attribution_gap",
    "top1_attribution_share",
    "top3_attribution_share",
    "normalized_attribution_entropy",
    "strongest_token",
    "strongest_shap",
]

unstable = unstable[
    unstable_columns
]

unstable_path = (
    OUTPUT_DIR
    / "unstable_attributions.csv"
)

unstable.to_csv(
    unstable_path,
    index=False
)

print(
    f"Unstable attribution instances: "
    f"{len(unstable):,}"
)

print(
    f"Saved: {unstable_path}"
)


# ============================================================
# HIGH CONCENTRATION ERRORS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONCENTRATION ERRORS")
print("=" * 70)

high_concentration = (
    profiles[
        profiles["correct"] == False
    ]
    .sort_values(
        "top1_attribution_share",
        ascending=False
    )
    .head(50)
)

print(
    high_concentration[
        [
            "sentence_id",
            "aspect",
            "true_polarity",
            "predicted_polarity",
            "top1_attribution_share",
            "top3_attribution_share",
            "context_attribution_ratio",
            "strongest_token",
            "strongest_shap",
        ]
    ].to_string(index=False)
)


# ============================================================
# SUMMARY JSON
# ============================================================

print("\n" + "=" * 70)
print("CREATING SUMMARY")
print("=" * 70)

correct_profiles = profiles[
    profiles["correct"] == True
]

incorrect_profiles = profiles[
    profiles["correct"] == False
]

summary_json = {
    "total_instances": int(
        len(profiles)
    ),
    "correct_instances": int(
        len(correct_profiles)
    ),
    "incorrect_instances": int(
        len(incorrect_profiles)
    ),
    "accuracy": float(
        len(correct_profiles)
        / len(profiles)
    ),
    "overall_mean_robustness_proxy": float(
        profiles["robustness_proxy"].mean()
    ),
    "overall_median_robustness_proxy": float(
        profiles["robustness_proxy"].median()
    ),
    "overall_mean_top1_attribution_share": float(
        profiles["top1_attribution_share"].mean()
    ),
    "overall_mean_top3_attribution_share": float(
        profiles["top3_attribution_share"].mean()
    ),
    "overall_mean_normalized_entropy": float(
        profiles[
            "normalized_attribution_entropy"
        ].mean()
    ),
    "overall_mean_context_attribution_ratio": float(
        profiles[
            "context_attribution_ratio"
        ].mean()
    ),
    "overall_mean_aspect_attribution_ratio": float(
        profiles[
            "aspect_attribution_ratio"
        ].mean()
    ),
    "overall_context_dominant_rate": float(
        profiles["context_dominant"].mean()
    ),
    "correct_mean_robustness_proxy": float(
        correct_profiles[
            "robustness_proxy"
        ].mean()
    ),
    "incorrect_mean_robustness_proxy": float(
        incorrect_profiles[
            "robustness_proxy"
        ].mean()
    ),
    "correct_mean_top1_attribution_share": float(
        correct_profiles[
            "top1_attribution_share"
        ].mean()
    ),
    "incorrect_mean_top1_attribution_share": float(
        incorrect_profiles[
            "top1_attribution_share"
        ].mean()
    ),
    "correct_mean_context_attribution_ratio": float(
        correct_profiles[
            "context_attribution_ratio"
        ].mean()
    ),
    "incorrect_mean_context_attribution_ratio": float(
        incorrect_profiles[
            "context_attribution_ratio"
        ].mean()
    ),
    "correct_context_dominant_rate": float(
        correct_profiles[
            "context_dominant"
        ].mean()
    ),
    "incorrect_context_dominant_rate": float(
        incorrect_profiles[
            "context_dominant"
        ].mean()
    ),
    "unstable_error_instances": int(
        len(unstable)
    ),
    "unstable_error_rate": float(
        len(unstable)
        / len(incorrect_profiles)
        if len(incorrect_profiles) > 0
        else 0.0
    ),
}

summary_json_path = (
    OUTPUT_DIR
    / "attribution_robustness_analysis_summary.json"
)

with open(
    summary_json_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary_json,
        f,
        indent=4,
    )

print(
    json.dumps(
        summary_json,
        indent=4
    )
)

print(
    f"\nSaved: {summary_json_path}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ATTRIBUTION ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nOutput directory:"
    f"\n{OUTPUT_DIR}"
)

print("\nGenerated files:")

for path in sorted(
    OUTPUT_DIR.iterdir()
):
    if path.is_file():
        print(
            f" - {path.name}"
        )
