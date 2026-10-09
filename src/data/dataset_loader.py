from pathlib import Path

import pandas as pd


LABELS = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(split: str) -> pd.DataFrame:
    """
    Load a processed MAMS split.

    Parameters
    ----------
    split:
        One of: train, val, test.
    """

    if split not in {"train", "val", "test"}:
        raise ValueError(
            "split must be one of: train, val, test"
        )

    path = (
        get_project_root()
        / "data"
        / "processed"
        / f"{split}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Processed split not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "sentence_id",
        "split",
        "sentence",
        "aspect",
        "polarity",
        "start",
        "end",
        "num_aspects",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {split}: {sorted(missing)}"
        )

    df = df.copy()

    df["label"] = df["polarity"].map(LABELS)

    if df["label"].isna().any():
        raise ValueError(
            f"Unknown polarity found in {split}"
        )

    return df


def load_all_splits():
    return {
        "train": load_split("train"),
        "val": load_split("val"),
        "test": load_split("test"),
    }


if __name__ == "__main__":

    splits = load_all_splits()

    for name, df in splits.items():
        print(
            f"{name:5s}: "
            f"{len(df):5d} instances | "
            f"{df['sentence_id'].nunique():4d} sentences"
        )