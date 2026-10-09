from __future__ import annotations

from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

import pandas as pd


VALID_POLARITIES = {"positive", "neutral", "negative"}


def parse_mams_xml(xml_path: str | Path, split: str) -> tuple[pd.DataFrame, dict]:
    """
    Parse a MAMS ABSA XML file.

    Each aspect annotation becomes one row.

    Returns
    -------
    df:
        Aspect-level dataframe.
    audit:
        Dataset validation information.
    """

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []

    sentence_count = 0
    aspect_count = 0

    polarity_counter = Counter()

    offset_mismatches = []
    invalid_offsets = []
    unknown_polarities = []
    empty_terms = []

    for sentence_idx, sentence_node in enumerate(root.findall("sentence")):

        text_node = sentence_node.find("text")

        if text_node is None or text_node.text is None:
            raise ValueError(
                f"Missing sentence text at sentence index {sentence_idx}"
            )

        text = text_node.text
        sentence_id = f"{split}_{sentence_idx + 1:06d}"

        aspect_terms_node = sentence_node.find("aspectTerms")

        if aspect_terms_node is None:
            aspect_nodes = []
        else:
            aspect_nodes = aspect_terms_node.findall("aspectTerm")

        num_aspects = len(aspect_nodes)
        sentence_count += 1

        for aspect_idx, aspect_node in enumerate(aspect_nodes):

            term = aspect_node.get("term")
            polarity = aspect_node.get("polarity")
            from_str = aspect_node.get("from")
            to_str = aspect_node.get("to")

            if term is None or term == "":
                empty_terms.append(
                    {
                        "sentence_id": sentence_id,
                        "aspect_index": aspect_idx,
                    }
                )
                continue

            if from_str is None or to_str is None:
                invalid_offsets.append(
                    {
                        "sentence_id": sentence_id,
                        "term": term,
                        "reason": "missing offset",
                    }
                )
                continue

            try:
                start = int(from_str)
                end = int(to_str)
            except ValueError:
                invalid_offsets.append(
                    {
                        "sentence_id": sentence_id,
                        "term": term,
                        "reason": "non-integer offset",
                    }
                )
                continue

            if start < 0 or end < 0 or start >= end or end > len(text):
                invalid_offsets.append(
                    {
                        "sentence_id": sentence_id,
                        "term": term,
                        "start": start,
                        "end": end,
                        "sentence_length": len(text),
                        "reason": "invalid range",
                    }
                )
                continue

            extracted_text = text[start:end]

            if extracted_text != term:
                offset_mismatches.append(
                    {
                        "sentence_id": sentence_id,
                        "term": term,
                        "extracted_text": extracted_text,
                        "start": start,
                        "end": end,
                    }
                )

            if polarity not in VALID_POLARITIES:
                unknown_polarities.append(
                    {
                        "sentence_id": sentence_id,
                        "term": term,
                        "polarity": polarity,
                    }
                )

            polarity_counter[polarity] += 1
            aspect_count += 1

            rows.append(
                {
                    "sentence_id": sentence_id,
                    "split": split,
                    "sentence": text,
                    "aspect": term,
                    "polarity": polarity,
                    "start": start,
                    "end": end,
                    "num_aspects": num_aspects,
                }
            )

    df = pd.DataFrame(rows)

    audit = {
        "split": split,
        "sentences": sentence_count,
        "aspect_terms": aspect_count,
        "positive": polarity_counter["positive"],
        "neutral": polarity_counter["neutral"],
        "negative": polarity_counter["negative"],
        "offset_mismatches": len(offset_mismatches),
        "invalid_offsets": len(invalid_offsets),
        "unknown_polarities": len(unknown_polarities),
        "empty_terms": len(empty_terms),
        "offset_mismatch_records": offset_mismatches,
        "invalid_offset_records": invalid_offsets,
        "unknown_polarity_records": unknown_polarities,
        "empty_term_records": empty_terms,
    }

    return df, audit


def print_audit(audit: dict) -> None:
    """Print a compact dataset audit."""

    print("=" * 60)
    print(f"SPLIT: {audit['split'].upper()}")
    print("=" * 60)

    print(f"Sentences       : {audit['sentences']}")
    print(f"Aspect terms    : {audit['aspect_terms']}")
    print(f"Positive        : {audit['positive']}")
    print(f"Neutral         : {audit['neutral']}")
    print(f"Negative        : {audit['negative']}")

    print()
    print("DATA QUALITY")
    print(f"Offset mismatch : {audit['offset_mismatches']}")
    print(f"Invalid offsets : {audit['invalid_offsets']}")
    print(f"Unknown labels  : {audit['unknown_polarities']}")
    print(f"Empty terms     : {audit['empty_terms']}")

    print()


def main() -> None:

    project_root = Path(__file__).resolve().parents[2]

    data_dir = project_root / "data" / "raw" / "mams"

    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": data_dir / "train.xml",
        "val": data_dir / "val.xml",
        "test": data_dir / "test.xml",
    }

    all_dataframes = []
    all_audits = []

    for split, xml_path in split_files.items():

        print(f"\nParsing {xml_path} ...")

        df, audit = parse_mams_xml(xml_path, split)

        print_audit(audit)

        all_dataframes.append(df)
        all_audits.append(audit)

        # Save individual split
        output_file = output_dir / f"{split}.csv"
        df.to_csv(output_file, index=False)

        print(f"Saved: {output_file}")

    combined_df = pd.concat(
        all_dataframes,
        ignore_index=True,
    )

    combined_file = output_dir / "mams_atsa_all.csv"
    combined_df.to_csv(combined_file, index=False)

    print("=" * 60)
    print("COMBINED DATASET")
    print("=" * 60)

    print(f"Rows            : {len(combined_df)}")
    print(f"Sentences       : {combined_df['sentence_id'].nunique()}")
    print()
    print("Polarity:")
    print(combined_df["polarity"].value_counts().to_string())

    print()
    print(f"Saved: {combined_file}")

    # Expected values from our verified dataset statistics.
    expected = {
        "train": {
            "sentences": 4297,
            "aspect_terms": 11186,
            "positive": 3380,
            "neutral": 5042,
            "negative": 2764,
        },
        "val": {
            "sentences": 500,
            "aspect_terms": 1332,
            "positive": 403,
            "neutral": 604,
            "negative": 325,
        },
        "test": {
            "sentences": 500,
            "aspect_terms": 1336,
            "positive": 400,
            "neutral": 607,
            "negative": 329,
        },
    }

    print()
    print("=" * 60)
    print("EXPECTED-VS-ACTUAL VALIDATION")
    print("=" * 60)

    validation_passed = True

    for audit in all_audits:

        split = audit["split"]
        exp = expected[split]

        checks = {
            "sentences": audit["sentences"] == exp["sentences"],
            "aspect_terms": audit["aspect_terms"] == exp["aspect_terms"],
            "positive": audit["positive"] == exp["positive"],
            "neutral": audit["neutral"] == exp["neutral"],
            "negative": audit["negative"] == exp["negative"],
        }

        print(f"\n{split.upper()}")

        for name, passed in checks.items():
            status = "PASS" if passed else "FAIL"

            print(
                f"{name:15s}: {status}"
                f" | expected={exp[name]}"
                f" | actual={audit[name]}"
            )

            if not passed:
                validation_passed = False

    print()

    if validation_passed:
        print("DATASET VALIDATION: PASSED")
    else:
        print("DATASET VALIDATION: FAILED")
        raise RuntimeError(
            "Parsed dataset does not match the expected statistics."
        )


if __name__ == "__main__":
    main()