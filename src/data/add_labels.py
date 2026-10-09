from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

POLARITY_TO_LABEL = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def process_split(split: str):

    path = DATA_DIR / f"{split}.csv"

    print(f"\nProcessing: {path}")

    df = pd.read_csv(path)

    if "polarity" not in df.columns:
        raise ValueError(
            f"{split}.csv does not contain a 'polarity' column."
        )

    # Convert polarity to numeric label
    df["label"] = (
        df["polarity"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(POLARITY_TO_LABEL)
    )

    # Check for unknown polarity values
    if df["label"].isna().any():

        bad_values = (
            df.loc[df["label"].isna(), "polarity"]
            .unique()
        )

        raise ValueError(
            f"Unknown polarity values in {split}.csv: "
            f"{bad_values}"
        )

    df["label"] = df["label"].astype(int)

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"{split}: {len(df):,} rows"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("Saved.")


def main():

    for split in ["train", "val", "test"]:
        process_split(split)

    print("\nAll dataset splits updated successfully.")


if __name__ == "__main__":
    main()

