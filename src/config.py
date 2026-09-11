"""Project paths and approved data-preparation / training settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Algerian_forest_fires_dataset_UPDATE.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "algerian_forest_fires_cleaned.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

# Saved training artifacts
MODEL_PIPELINE_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"
EVALUATION_METRICS_PATH = METRICS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_FIG_PATH = FIGURES_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_FIG_PATH = FIGURES_DIR / "feature_importance.png"

# Train/test split settings
# 20% hold-out keeps most of the small sample for training while still
# providing a stratified held-out test set for unbiased evaluation.
TEST_SIZE = 0.20

# Random Forest hyperparameters (chosen a priori; not tuned on the test set)
RF_HYPERPARAMETERS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": None,
    "n_jobs": -1,
}

# Optional stability check on the TRAINING split only (not the held-out test set)
CV_FOLDS = 5

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
