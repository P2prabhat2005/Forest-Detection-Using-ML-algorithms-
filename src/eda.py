"""Generate EDA figures from the cleaned processed dataset.

No model training is performed in this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, NUMERIC_FEATURES, PROCESSED_DATA_PATH, TARGET_COLUMN


def _ensure_figs_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def load_processed(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run: python -m src.prepare_data"
        )
    return pd.read_csv(path)


def plot_class_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Preserve original label strings from the cleaned dataset
    counts = df[TARGET_COLUMN].value_counts().reindex(["not fire", "fire"])
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#2f6f4e")
    ax.set_title("Class Distribution (Fire vs Not Fire)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, float(v) + 1, str(int(v)), ha="center")
    fig.tight_layout()
    path = out_dir / "class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_histograms(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = NUMERIC_FEATURES
    n = len(cols)
    nrows = 3
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 9))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], ax=axes[i], kde=True, color="#3b6ea5")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric Feature Distributions", y=1.01)
    fig.tight_layout()
    path = out_dir / "feature_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = df[NUMERIC_FEATURES].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.tight_layout()
    path = out_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_features_by_class(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = [c for c in NUMERIC_FEATURES if c != "month"]
    melted = df.melt(
        id_vars=[TARGET_COLUMN],
        value_vars=cols,
        var_name="feature",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.boxplot(data=melted, x="feature", y="value", hue=TARGET_COLUMN, ax=ax)
    ax.set_title("Feature Values by Class (Fire vs Not Fire)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = out_dir / "features_by_class_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_region_class_counts(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct = pd.crosstab(df["Region"], df[TARGET_COLUMN])
    ct.plot(kind="bar", ax=ax, color=["#4c78a8", "#c44e52"])
    ax.set_title("Fire vs Not Fire Counts by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Count")
    ax.legend(title="Class")
    fig.tight_layout()
    path = out_dir / "region_class_counts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_month_class_counts(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct = pd.crosstab(df["month"], df[TARGET_COLUMN])
    ct.plot(kind="bar", ax=ax, color=["#4c78a8", "#c44e52"])
    ax.set_title("Fire vs Not Fire Counts by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.legend(title="Class")
    fig.tight_layout()
    path = out_dir / "month_class_counts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_eda() -> dict:
    sns.set_theme(style="whitegrid")
    out_dir = _ensure_figs_dir()
    df = load_processed()

    outputs = {
        "class_distribution": str(plot_class_distribution(df, out_dir)),
        "feature_distributions": str(plot_feature_histograms(df, out_dir)),
        "correlation_heatmap": str(plot_correlation_heatmap(df, out_dir)),
        "features_by_class": str(plot_features_by_class(df, out_dir)),
        "region_class_counts": str(plot_region_class_counts(df, out_dir)),
        "month_class_counts": str(plot_month_class_counts(df, out_dir)),
        "n_rows": int(len(df)),
        "class_counts": df[TARGET_COLUMN].value_counts().to_dict(),
    }
    return outputs


if __name__ == "__main__":
    result = run_eda()
    print("EDA figures written:")
    for k, v in result.items():
        if k.endswith("_counts") or k == "n_rows":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
