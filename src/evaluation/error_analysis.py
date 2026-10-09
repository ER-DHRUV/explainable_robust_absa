from __future__ import annotations


import json
from pathlib import Path

import pandas as pd

# ============================================================

# CONFIGURATION

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
PROJECT_ROOT
/ "results"
/ "transformer"
/ "distilbert_baseline"
/ "test_predictions.csv"
)

OUTPUT_DIR = (
PROJECT_ROOT
/ "results"
/ "transformer"
/ "distilbert_baseline"
/ "error_analysis"
)

OUTPUT_DIR.mkdir(
parents=True,
exist_ok=True,
)

LABELS = [
"negative",
"neutral",
"positive",
]

# ============================================================

# DATA LOADING

# ============================================================

def load_predictions() -> pd.DataFrame:


   if not PREDICTIONS_FILE.exists():
       raise FileNotFoundError(
           f"Prediction file not found:\n{PREDICTIONS_FILE}"
       )
   
   df = pd.read_csv(PREDICTIONS_FILE)
   
   required_columns = {
       "sentence_id",
       "sentence",
       "aspect",
       "polarity",
       "true_label",
       "predicted_label",
       "true_polarity",
       "predicted_polarity",
       "correct",
   }
   
   missing = required_columns - set(df.columns)
   
   if missing:
       raise ValueError(
           f"Missing required columns: {sorted(missing)}"
       )
   
   # Normalize boolean column in case CSV stores it as text.
   if df["correct"].dtype != bool:
       df["correct"] = (
           df["correct"]
           .astype(str)
           .str.lower()
           .map({
               "true": True,
               "false": False,
           })
       )
   
   return df


# ============================================================

# BASIC SUMMARY

# ============================================================

def basic_summary(df: pd.DataFrame) -> dict:


    total = len(df)
    correct = int(df["correct"].sum())
    incorrect = total - correct
    
    return {
        "total_instances": total,
        "correct_instances": correct,
        "incorrect_instances": incorrect,
        "accuracy": (
            correct / total
            if total > 0
            else 0.0
        ),
        "error_rate": (
            incorrect / total
            if total > 0
            else 0.0
        ),
    }


# ============================================================

# CONFUSION PAIRS

# ============================================================

def confusion_pair_analysis(df: pd.DataFrame,) -> pd.DataFrame:

   
   errors = df[
       df["true_polarity"]
       != df["predicted_polarity"]
   ].copy()
   
   result = (
       errors
       .groupby(
           [
               "true_polarity",
               "predicted_polarity",
           ],
           dropna=False,
       )
       .size()
       .reset_index(name="count")
       .sort_values(
           "count",
           ascending=False,
       )
   )
   
   if len(result) > 0:
       result["error_percentage"] = (
           result["count"]
           / len(errors)
           * 100
       )
   
   return result


# ============================================================

# ERRORS BY TRUE POLARITY

# ============================================================

def errors_by_true_polarity(df: pd.DataFrame,) -> pd.DataFrame:


    result = (
        df.groupby("true_polarity")
        .agg(
            total_instances=("correct", "size"),
            errors=(
                "correct",
                lambda x: int((~x).sum()),
            ),
        )
        .reindex(LABELS)
        .reset_index()
    )
    
    result["correct"] = (
        result["total_instances"]
        - result["errors"]
    )
    
    result["error_rate"] = (
        result["errors"]
        / result["total_instances"]
        * 100
    )
    
    return result


# ============================================================

# ERRORS BY PREDICTED POLARITY

# ============================================================

def errors_by_predicted_polarity(df: pd.DataFrame,) -> pd.DataFrame:

     
     errors = df[~df["correct"]].copy()
     
     result = (
         errors
         .groupby("predicted_polarity")
         .size()
         .reindex(
             LABELS,
             fill_value=0,
         )
         .reset_index(name="errors")
     )
     
     return result


# ============================================================

# ASPECT-LEVEL ERROR ANALYSIS

# ============================================================

def aspect_error_analysis(df: pd.DataFrame,) -> pd.DataFrame:

     
     result = (
         df.groupby("aspect")
         .agg(
             total_instances=("correct", "size"),
             errors=(
                 "correct",
                 lambda x: int((~x).sum()),
             ),
         )
         .reset_index()
     )
     
     result["correct"] = (
         result["total_instances"]
         - result["errors"]
     )
     
     result["error_rate"] = (
         result["errors"]
         / result["total_instances"]
         * 100
     )
     
     result = result.sort_values(
         [
             "errors",
             "error_rate",
         ],
         ascending=False,
     )
     
     return result


# ============================================================

# SENTENCE-LEVEL ERROR ANALYSIS

# ============================================================

def sentence_error_analysis(df: pd.DataFrame,) -> pd.DataFrame:


    def total_aspects(group):
        return len(group)
    
    def error_count(group):
        return int((~group["correct"]).sum())
        
    result = (
        df.groupby(
            [
                "sentence_id",
                "sentence",
            ],
            sort=False,
        )
        .apply(
            lambda group: pd.Series(
                {
                    "num_aspects": total_aspects(group),
                    "num_errors": error_count(group),
                    "num_correct": int(
                        group["correct"].sum()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    
    result["sentence_has_error"] = (
        result["num_errors"] > 0
    )
    
    result["all_aspects_correct"] = (
        result["num_errors"] == 0
    )
    
    result["all_aspects_wrong"] = (
        result["num_errors"]
        == result["num_aspects"]
    )
    
    result["mixed_correct_incorrect"] = (
        (result["num_errors"] > 0)
        & (result["num_correct"] > 0)
    )
    
    result["sentence_error_rate"] = (
        result["num_errors"]
        / result["num_aspects"]
        * 100
    )
    
    return result


# ============================================================

# ERROR-RATE BY ASPECT DENSITY

# ============================================================

def aspect_density_analysis(df: pd.DataFrame,) -> pd.DataFrame:

    
    sentence_stats = (
        df.groupby("sentence_id")
        .agg(
            num_aspects=("aspect", "size"),
            errors=(
                "correct",
                lambda x: int((~x).sum()),
            ),
            total_instances=(
                "correct",
                "size",
            ),
        )
        .reset_index()
    )
    
    result = (
        sentence_stats
        .groupby("num_aspects")
        .agg(
            sentences=("sentence_id", "size"),
            total_instances=(
                "total_instances",
                "sum",
            ),
            errors=("errors", "sum"),
        )
        .reset_index()
    )
    
    result["error_rate"] = (
        result["errors"]
        / result["total_instances"]
        * 100
    )
    
    return result.sort_values("num_aspects")


# ============================================================

# MULTI-ASPECT ERROR ANALYSIS

# ============================================================

def multi_aspect_summary(df: pd.DataFrame,sentence_stats: pd.DataFrame,) -> dict:

    
    multi = sentence_stats[
        sentence_stats["num_aspects"] > 1
    ]
    
    return {
        "multi_aspect_sentences": int(len(multi)),
        "sentences_with_any_error": int(
            multi["sentence_has_error"].sum()
        ),
        "all_aspects_correct": int(
            multi["all_aspects_correct"].sum()
        ),
        "all_aspects_wrong": int(
            multi["all_aspects_wrong"].sum()
        ),
        "mixed_correct_incorrect": int(
            multi["mixed_correct_incorrect"].sum()
        ),
    }


# ============================================================

# SAVE ERROR INSTANCE TABLE

# ============================================================

def save_error_instances(
df: pd.DataFrame,
) -> None:


    errors = df[~df["correct"]].copy()
    
    errors = errors[
        [
            "sentence_id",
            "sentence",
            "aspect",
            "polarity",
            "true_polarity",
            "predicted_polarity",
        ]
    ]
    
    errors.to_csv(
        OUTPUT_DIR / "all_errors.csv",
        index=False,
    )


# ============================================================

# SAVE MIXED SENTENCES

# ============================================================

def save_mixed_sentences(
df: pd.DataFrame,
sentence_stats: pd.DataFrame,
) -> None:


   mixed_ids = set(
       sentence_stats.loc[
           sentence_stats["mixed_correct_incorrect"],
           "sentence_id",
       ]
   )
   
   mixed = df[
       df["sentence_id"].isin(mixed_ids)
   ].copy()
   
   mixed.to_csv(
       OUTPUT_DIR / "mixed_correct_incorrect.csv",
       index=False,
   )


# ============================================================

# SAVE MOST DIFFICULT ASPECTS

# ============================================================

def save_difficult_aspects(
aspect_stats: pd.DataFrame,
) -> None:


    difficult = aspect_stats[
        aspect_stats["total_instances"] >= 3
    ].copy()
    
    difficult = difficult.sort_values(
        [
            "error_rate",
            "errors",
        ],
        ascending=False,
    )
    
    difficult.head(100).to_csv(
        OUTPUT_DIR / "most_error_prone_aspects.csv",
        index=False,
    )


# ============================================================

# PRINT REPORT

# ============================================================

def print_report(
summary: dict,
confusion: pd.DataFrame,
true_polarity: pd.DataFrame,
predicted_polarity: pd.DataFrame,
aspect_stats: pd.DataFrame,
sentence_stats: pd.DataFrame,
density_stats: pd.DataFrame,
multi_summary: dict,
) -> None:

     
     print("=" * 70)
     print("DISTILBERT ERROR ANALYSIS")
     print("=" * 70)
     
     print()
     print("DATASET")
     print("-" * 70)
     
     print(
         f"Total instances : "
         f"{summary['total_instances']:,}"
     )
     
     print(
         f"Correct         : "
         f"{summary['correct_instances']:,}"
     )
     
     print(
         f"Incorrect       : "
         f"{summary['incorrect_instances']:,}"
     )
     
     print(
         f"Accuracy        : "
         f"{summary['accuracy']:.4f}"
     )
     
     print(
         f"Error rate      : "
         f"{summary['error_rate']:.4f}"
     )
     
     print()
     print("CONFUSION PAIRS")
     print("-" * 70)
     
     if len(confusion) == 0:
         print("No errors.")
     else:
         for _, row in confusion.iterrows():
             print(
                 f"{row['true_polarity']:>8} -> "
                 f"{row['predicted_polarity']:<8} "
                 f"{int(row['count']):>4} "
                 f"({row['error_percentage']:.2f}%)"
             )
     
     print()
     print("ERRORS BY TRUE POLARITY")
     print("-" * 70)
     
     for _, row in true_polarity.iterrows():
         print(
             f"{row['true_polarity']:>8}: "
             f"{int(row['errors']):>4} errors / "
             f"{int(row['total_instances']):>4} "
             f"({row['error_rate']:.2f}%)"
         )
     
     print()
     print("ERRORS BY PREDICTED POLARITY")
     print("-" * 70)
     
     for _, row in predicted_polarity.iterrows():
         print(
             f"{row['predicted_polarity']:>8}: "
             f"{int(row['errors']):>4}"
         )
     
     print()
     print("ASPECT DENSITY")
     print("-" * 70)
     
     for _, row in density_stats.iterrows():
         print(
             f"{int(row['num_aspects']):>2} aspects/sentence: "
             f"{int(row['sentences']):>4} sentences | "
             f"{int(row['errors']):>4} errors / "
             f"{int(row['total_instances']):>4} instances | "
             f"error rate={row['error_rate']:.2f}%"
         )
     
     print()
     print("MULTI-ASPECT SENTENCE ANALYSIS")
     print("-" * 70)
     
     print(
         f"Multi-aspect sentences     : "
         f"{multi_summary['multi_aspect_sentences']:,}"
     )
     
     print(
         f"Sentences with any error   : "
         f"{multi_summary['sentences_with_any_error']:,}"
     )
     
     print(
         f"All aspects correct        : "
         f"{multi_summary['all_aspects_correct']:,}"
     )
     
     print(
         f"All aspects wrong          : "
         f"{multi_summary['all_aspects_wrong']:,}"
     )
     
     print(
         f"Mixed correct/incorrect    : "
         f"{multi_summary['mixed_correct_incorrect']:,}"
     )
     
     print()
     print("TOP ERROR-PRONE ASPECTS")
     print("-" * 70)
     
     difficult = aspect_stats[
         aspect_stats["total_instances"] >= 3
     ].head(20)
     
     for _, row in difficult.iterrows():
         print(
             f"{str(row['aspect'])[:40]:<40} "
             f"instances={int(row['total_instances']):>3} "
             f"errors={int(row['errors']):>3} "
             f"rate={row['error_rate']:>6.2f}%"
         )


# ============================================================

# MAIN

# ============================================================

def main():


  print()
  print("Loading predictions...")
  
  df = load_predictions()
  
  summary = basic_summary(df)
  
  confusion = confusion_pair_analysis(df)
  
  true_polarity = errors_by_true_polarity(df)
  
  predicted_polarity = (
      errors_by_predicted_polarity(df)
  )
  
  aspect_stats = aspect_error_analysis(df)
  
  sentence_stats = sentence_error_analysis(df)
  
  density_stats = aspect_density_analysis(df)
  
  multi_summary = multi_aspect_summary(
      df,
      sentence_stats,
  )

  # --------------------------------------------------------
  # Save tables
  # --------------------------------------------------------
  
  confusion.to_csv(
      OUTPUT_DIR / "confusion_pairs.csv",
      index=False,
  )
  
  true_polarity.to_csv(
      OUTPUT_DIR / "errors_by_true_polarity.csv",
      index=False,
  )
  
  predicted_polarity.to_csv(
      OUTPUT_DIR / "errors_by_predicted_polarity.csv",
      index=False,
  )
  
  aspect_stats.to_csv(
      OUTPUT_DIR / "errors_by_aspect.csv",
      index=False,
  )
  
  sentence_stats.to_csv(
      OUTPUT_DIR / "errors_by_sentence.csv",
      index=False,
  )
  
  density_stats.to_csv(
      OUTPUT_DIR / "errors_by_aspect_density.csv",
      index=False,
  )
  
  save_error_instances(df)
  
  save_mixed_sentences(
      df,
      sentence_stats,
  )
  
  save_difficult_aspects(
      aspect_stats,
  )
  
  # --------------------------------------------------------
  # JSON summary
  # --------------------------------------------------------
  
  json_summary = {
      "model": "distilbert-base-uncased",
      "split": "test",
      "prediction_file": str(
          PREDICTIONS_FILE
      ),
      "summary": summary,
      "confusion_pairs": (
          confusion.to_dict(
              orient="records"
          )
      ),
      "errors_by_true_polarity": (
          true_polarity.to_dict(
              orient="records"
          )
      ),
      "errors_by_predicted_polarity": (
          predicted_polarity.to_dict(
              orient="records"
          )
      ),
      "multi_aspect_analysis": multi_summary,
  }
  
  with open(
      OUTPUT_DIR / "error_summary.json",
      "w",
      encoding="utf-8",
  ) as f:
      json.dump(
          json_summary,
          f,
          indent=4,
      )
  
  # --------------------------------------------------------
  # Report
  # --------------------------------------------------------
  
  print_report(
      summary,
      confusion,
      true_polarity,
      predicted_polarity,
      aspect_stats,
      sentence_stats,
      density_stats,
      multi_summary,
  )
  
  print()
  print("=" * 70)
  print("ERROR ANALYSIS COMPLETE")
  print("=" * 70)
  
  print(
      f"Output directory:\n{OUTPUT_DIR}"
  )
  
  print()
  print("Generated files:")
  
  for path in sorted(OUTPUT_DIR.iterdir()):
      if path.is_file():
          print(f"  - {path.name}")


if __name__ == "__main__":
    main()

