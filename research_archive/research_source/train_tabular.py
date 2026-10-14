"""Train a Random Forest on the Algerian Forest Fires dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "tabular" / "algerian_forest_fires.csv"
MODEL_PATH = ROOT / "models" / "tabular_rf.joblib"
METRICS_PATH = ROOT / "models" / "tabular_metrics.json"
CM_PATH = ROOT / "models" / "tabular_confusion_matrix.png"

FEATURES = [
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


def load_algerian_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/download_data.py"
        )

    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: list[str] | None = None
    rows: list[list[str]] = []

    for line in raw:
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if "region" in lower and "dataset" in lower:
            continue
        cols = [c.strip() for c in line.split(",")]
        if cols[0].lower() == "day":
            header = cols
            continue
        if header is None:
            continue
        # One published row is missing a comma between DC and ISI.
        if len(cols) == len(header) - 1:
            repaired = cols[:9] + [cols[9][: cols[9].find(".") + 2], cols[9][cols[9].find(".") + 2 :]] + cols[10:]
            if len(repaired) == len(header):
                cols = repaired
        if len(cols) != len(header):
            continue
        rows.append(cols)

    if header is None or not rows:
        raise ValueError(f"Could not parse forest-fire CSV at {path}")

    df = pd.DataFrame(rows, columns=header)
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    class_col = next(c for c in work.columns if c.lower().startswith("class"))
    work[class_col] = (
        work[class_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    work["label"] = work[class_col].str.contains("not").map({True: 0, False: 1})

    for col in FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=FEATURES + ["label"])
    return work[FEATURES], work["label"].astype(int)


def train() -> None:
    X, y = prepare_xy(load_algerian_csv(CSV_PATH))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=["not fire", "fire"], output_dict=True
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": FEATURES}, MODEL_PATH)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "confusion_matrix": cm,
        "labels": ["not fire", "fire"],
        "classification_report": report,
        "features": FEATURES,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="YlOrRd")
    ax.set_xticks([0, 1], ["not fire", "fire"])
    ax.set_yticks([0, 1], ["not fire", "fire"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=120)
    plt.close(fig)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Accuracy={acc:.3f}  F1={f1:.3f}")
    print(classification_report(y_test, y_pred, target_names=["not fire", "fire"]))


if __name__ == "__main__":
    try:
        train()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
