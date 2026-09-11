"""Project paths and approved data-preparation settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Algerian_forest_fires_dataset_UPDATE.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "algerian_forest_fires_cleaned.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Raw CSV header fields after stripping whitespace
RAW_COLUMNS = [
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
    "Classes",
]

# Approved primary PBL feature set
NUMERIC_FEATURES = [
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "month",
]
CATEGORICAL_FEATURES = ["Region"]
# Exact approved primary feature order
FEATURE_COLUMNS = [
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "Region",
    "month",
]

# Modeling choice for the primary PBL model (not a claim of universal uselessness)
EXCLUDED_FROM_PRIMARY_MODEL = ["FWI", "BUI", "day", "year"]

TARGET_COLUMN = "Classes"
TARGET_ENCODED_COLUMN = "Classes_encoded"
TARGET_MAPPING = {
    "not fire": 0,
    "fire": 1,
}
EXPECTED_TARGET_LABELS = set(TARGET_MAPPING.keys())

REGION_BEJAIA = "Bejaia"
REGION_SIDI_BEL_ABBES = "Sidi-Bel-Abbes"
EXPECTED_REGIONS = {REGION_BEJAIA, REGION_SIDI_BEL_ABBES}

RANDOM_STATE = 42
