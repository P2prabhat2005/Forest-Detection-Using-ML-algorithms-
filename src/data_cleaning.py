"""Clean the raw UCI Algerian Forest Fires CSV into a modeling-ready table.

Important:
- Never modifies data/raw/
- Drops the malformed row (does not repair it)
- Does not invent values
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    EXCLUDED_FROM_PRIMARY_MODEL,
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    PROCESSED_DIR,
    RAW_COLUMNS,
    RAW_DATA_PATH,
    REGION_BEJAIA,
    REGION_SIDI_BEL_ABBES,
    TARGET_COLUMN,
    TARGET_ENCODED_COLUMN,
    TARGET_MAPPING,
)
from src.validation import (
    DataValidationError,
    validate_cleaned_frame,
    validate_target_labels,
)


def _is_region_title(line: str) -> bool:
    lowered = line.strip().lower()
    return "region" in lowered and "dataset" in lowered and "," not in line


def _is_header_line(line: str) -> bool:
    stripped = line.strip().lower().replace(" ", "")
    return stripped.startswith("day,month,year,temperature")


def parse_raw_records(raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Parse the messy raw CSV into row records with a derived Region column."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    lines = raw_path.read_text(encoding="utf-8", errors="replace").splitlines()
    current_region = None
    records: list[dict] = []
    parse_notes: list[str] = []

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if _is_region_title(line):
            lowered = line.lower()
            if "bejaia" in lowered:
                current_region = REGION_BEJAIA
            elif "sidi" in lowered:
                current_region = REGION_SIDI_BEL_ABBES
            else:
                raise DataValidationError(
                    f"Unrecognized region title at line {line_no}: {line!r}"
                )
            continue
        if _is_header_line(line):
            continue
        if current_region is None:
            raise DataValidationError(
                f"Data row encountered before a region title at line {line_no}."
            )

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(RAW_COLUMNS):
            parse_notes.append(
                f"line {line_no}: expected {len(RAW_COLUMNS)} fields, got {len(parts)}"
            )

        if len(parts) < len(RAW_COLUMNS):
            parts = parts + [""] * (len(RAW_COLUMNS) - len(parts))
        elif len(parts) > len(RAW_COLUMNS):
            parts = parts[: len(RAW_COLUMNS)]

        row = dict(zip(RAW_COLUMNS, parts))
        row["Region"] = current_region
        row["_source_line"] = line_no
        records.append(row)

    if not records:
        raise DataValidationError("No data rows parsed from raw CSV.")

    df = pd.DataFrame.from_records(records)
    df.attrs["parse_notes"] = parse_notes
    return df


def _normalize_classes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def identify_malformed_mask(df: pd.DataFrame) -> pd.Series:
    """Flag rows that cannot be used without inventing/repairing values.

    Approved policy: DROP these rows (do not repair).
    """
    classes = _normalize_classes(df[TARGET_COLUMN])
    bad_target = ~classes.isin(TARGET_MAPPING.keys())

    check_cols = [
        "day",
        "month",
        "year",
        "Temperature",
        "RH",
        "Ws",
        "Rain",
        "FFMC",
        "DMC",
        "DC",
        "ISI",
        "BUI",
        "FWI",
    ]
    bad_numeric = pd.Series(False, index=df.index)
    for col in check_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad_numeric |= coerced.isna()

    return bad_target | bad_numeric


def clean_raw_dataset(raw_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Return cleaned modeling table and a cleaning summary dict."""
    parsed = parse_raw_records(raw_path)
    n_parsed = len(parsed)

    malformed_mask = identify_malformed_mask(parsed)
    malformed_rows = parsed.loc[malformed_mask].copy()
    kept = parsed.loc[~malformed_mask].copy()

    dropped_detail = []
    if not malformed_rows.empty:
        dropped_detail = malformed_rows[
            ["_source_line", "Region", "DC", "FWI", TARGET_COLUMN]
        ].to_dict(orient="records")

    kept[TARGET_COLUMN] = _normalize_classes(kept[TARGET_COLUMN])
    validate_target_labels(kept[TARGET_COLUMN], "after cleaning")

    for col in [
        "day",
        "month",
        "year",
        "Temperature",
        "RH",
        "Ws",
        "Rain",
        "FFMC",
        "DMC",
        "DC",
        "ISI",
        "BUI",
        "FWI",
    ]:
        kept[col] = pd.to_numeric(kept[col], errors="raise")

    kept[TARGET_ENCODED_COLUMN] = kept[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)

    cleaned = kept[FEATURE_COLUMNS + [TARGET_COLUMN, TARGET_ENCODED_COLUMN]].copy()
    cleaned = cleaned.reset_index(drop=True)
    validate_cleaned_frame(cleaned)

    summary = {
        "raw_path": str(raw_path),
        "rows_parsed": n_parsed,
        "rows_dropped_malformed": int(malformed_mask.sum()),
        "dropped_row_details": dropped_detail,
        "rows_kept": int(len(cleaned)),
        "features_kept": list(FEATURE_COLUMNS),
        "features_excluded_primary_model": list(EXCLUDED_FROM_PRIMARY_MODEL),
        "exclusion_rationale": (
            "FWI and BUI are excluded from the primary PBL model because they are "
            "composite/redundant with other fire-weather variables, giving a clearer "
            "and less redundant feature set. day is excluded to reduce calendar "
            "memorization risk on a tiny single-year sample. year is excluded because "
            "it is constant (2012). This is a modeling choice, not a claim that "
            "FWI/BUI are universally useless."
        ),
        "target_mapping": dict(TARGET_MAPPING),
        "class_counts": cleaned[TARGET_COLUMN].value_counts().to_dict(),
        "class_counts_encoded": cleaned[TARGET_ENCODED_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict(),
        "region_counts": cleaned["Region"].value_counts().to_dict(),
        "parse_notes": parsed.attrs.get("parse_notes", []),
    }
    return cleaned, summary


def save_cleaned_dataset(
    cleaned: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_PATH,
) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    validate_cleaned_frame(cleaned)
    cleaned.to_csv(output_path, index=False)
    return output_path


def prepare_processed_dataset(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> tuple[pd.DataFrame, dict, Path]:
    cleaned, summary = clean_raw_dataset(raw_path)
    path = save_cleaned_dataset(cleaned, output_path)
    summary["processed_path"] = str(path)
    return cleaned, summary, path


if __name__ == "__main__":
    cleaned_df, info, out = prepare_processed_dataset()
    print("Processed dataset written to:", out)
    print("Shape:", cleaned_df.shape)
    print("Class counts:", info["class_counts"])
    print("Dropped malformed rows:", info["rows_dropped_malformed"])
    if info["dropped_row_details"]:
        print("Dropped row detail:", info["dropped_row_details"])
