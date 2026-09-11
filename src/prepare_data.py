"""CLI entry point: clean raw data and write data/processed/ output."""

from __future__ import annotations

import json

from src.config import PROCESSED_DIR
from src.data_cleaning import prepare_processed_dataset


def main() -> None:
    cleaned, summary, path = prepare_processed_dataset()
    summary_path = PROCESSED_DIR / "cleaning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print("DATA PREPARATION COMPLETE (no model training)")
    print("=" * 60)
    print(f"Processed CSV : {path}")
    print(f"Summary JSON  : {summary_path}")
    print(f"Shape         : {cleaned.shape}")
    print(f"Features      : {summary['features_kept']}")
    print(f"Class counts  : {summary['class_counts']}")
    print(f"Dropped rows  : {summary['rows_dropped_malformed']}")
    if summary["dropped_row_details"]:
        print(f"Drop detail   : {summary['dropped_row_details']}")


if __name__ == "__main__":
    main()
