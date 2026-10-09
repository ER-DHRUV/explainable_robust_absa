from pathlib import Path
import pandas as pd


def main():

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"

    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    datasets = {
        "train": train,
        "val": val,
        "test": test,
    }

    print("=" * 70)
    print("MAMS DATASET AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Basic statistics
    # ---------------------------------------------------------

    for name, df in datasets.items():

        print(f"\n{name.upper()}")

        print(f"Aspect instances : {len(df):,}")
        print(f"Sentences        : {df['sentence_id'].nunique():,}")
        print(
            f"Avg aspects/sent : "
            f"{len(df) / df['sentence_id'].nunique():.3f}"
        )

        print("\nPolarity distribution:")

        counts = df["polarity"].value_counts()
        percentages = df["polarity"].value_counts(normalize=True) * 100

        for label in ["positive", "neutral", "negative"]:
            print(
                f"  {label:8s}: "
                f"{counts.get(label, 0):5d} "
                f"({percentages.get(label, 0):6.2f}%)"
            )

    # ---------------------------------------------------------
    # 2. Aspect density
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ASPECT DENSITY")
    print("=" * 70)

    for name, df in datasets.items():

        sentence_density = (
            df.groupby("sentence_id")
            .size()
            .value_counts()
            .sort_index()
        )

        print(f"\n{name.upper()}")

        for aspect_count, sentence_count in sentence_density.items():
            print(
                f"  {aspect_count:2d} aspect(s): "
                f"{sentence_count:4d} sentence(s)"
            )

    # ---------------------------------------------------------
    # 3. Cross-split sentence leakage
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CROSS-SPLIT SENTENCE LEAKAGE")
    print("=" * 70)

    sentence_sets = {
        name: set(df["sentence"].astype(str))
        for name, df in datasets.items()
    }

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    leakage_found = False

    for a, b in pairs:

        overlap = sentence_sets[a] & sentence_sets[b]

        print(
            f"{a:5s} ∩ {b:5s}: "
            f"{len(overlap)} sentence(s)"
        )

        if overlap:
            leakage_found = True

            print("  Example:")
            print(" ", next(iter(overlap))[:200])

    # ---------------------------------------------------------
    # 4. Sentence-ID overlap
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SENTENCE-ID OVERLAP")
    print("=" * 70)

    id_sets = {
        name: set(df["sentence_id"])
        for name, df in datasets.items()
    }

    for a, b in pairs:

        overlap = id_sets[a] & id_sets[b]

        print(
            f"{a:5s} ∩ {b:5s}: "
            f"{len(overlap)} sentence ID(s)"
        )

    # ---------------------------------------------------------
    # 5. Multi-aspect polarity composition
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MULTI-ASPECT SENTENCE POLARITY ANALYSIS")
    print("=" * 70)

    combined = pd.concat(
        datasets.values(),
        ignore_index=True
    )

    sentence_polarities = (
        combined
        .groupby(["split", "sentence_id"])["polarity"]
        .apply(set)
        .reset_index(name="polarities")
    )

    sentence_polarities["num_polarities"] = (
        sentence_polarities["polarities"].apply(len)
    )

    for split in ["train", "val", "test"]:

        subset = sentence_polarities[
            sentence_polarities["split"] == split
        ]

        print(f"\n{split.upper()}")

        print(
            "Sentences with 1 polarity :",
            (subset["num_polarities"] == 1).sum()
        )

        print(
            "Sentences with 2 polarities:",
            (subset["num_polarities"] == 2).sum()
        )

        print(
            "Sentences with 3 polarities:",
            (subset["num_polarities"] == 3).sum()
        )

    # ---------------------------------------------------------
    # 6. Final verdict
    # ---------------------------------------------------------

    print("\n" + "=" * 70)

    if leakage_found:
        print("WARNING: CROSS-SPLIT SENTENCE OVERLAP DETECTED")
    else:
        print("CROSS-SPLIT SENTENCE LEAKAGE: NONE DETECTED")

    print("=" * 70)


if __name__ == "__main__":
    main()