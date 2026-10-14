"""Download tabular data and optionally create a synthetic image training set."""

from __future__ import annotations

import argparse
import random
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
TABULAR_DIR = ROOT / "data" / "tabular"
IMAGES_TRAIN = ROOT / "data" / "images" / "train"
IMAGES_VAL = ROOT / "data" / "images" / "val"
SAMPLES = ROOT / "data" / "images" / "samples"

UCI_CSV = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00547/Algerian_forest_fires_dataset_UPDATE.csv"
)

IMAGE_DATASET_NOTES = """
Image dataset (for real photos)
-------------------------------
Place fire / no-fire photos in ImageFolder layout:

  data/images/train/fire/
  data/images/train/no_fire/
  data/images/val/fire/
  data/images/val/no_fire/

Public sources:
  - Kaggle: "FIRE Dataset" (https://www.kaggle.com/datasets/phylake1337/fire-dataset)
  - Kaggle: "Forest Fire Dataset" image collections
  - FLAME: Aerial forest fire imagery (research use)

Then run:  python src/train_image.py

This script can also generate synthetic color images for a local smoke test:
  python scripts/download_data.py --synthetic-images
"""


def download_tabular() -> Path:
    TABULAR_DIR.mkdir(parents=True, exist_ok=True)
    dest = TABULAR_DIR / "algerian_forest_fires.csv"
    print(f"Downloading Algerian Forest Fires CSV -> {dest}")
    urllib.request.urlretrieve(UCI_CSV, dest)
    print(f"Saved {dest.stat().st_size} bytes")
    return dest


def _blob(draw: ImageDraw.ImageDraw, color: tuple[int, int, int], size: int) -> None:
    for _ in range(random.randint(8, 18)):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        r = random.randint(12, 60)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def make_fire_image(size: int = 224) -> Image.Image:
    img = Image.new("RGB", (size, size), (20, 12, 8))
    draw = ImageDraw.Draw(img)
    for color in ((40, 25, 15), (90, 35, 10), (200, 80, 15), (255, 160, 30), (255, 230, 80)):
        _blob(draw, color, size)
    return img.filter(ImageFilter.GaussianBlur(radius=1.2))


def make_nofire_image(size: int = 224) -> Image.Image:
    img = Image.new("RGB", (size, size), (18, 40, 16))
    draw = ImageDraw.Draw(img)
    for color in ((12, 55, 22), (34, 90, 40), (70, 130, 55), (100, 150, 80), (40, 70, 110)):
        _blob(draw, color, size)
    return img.filter(ImageFilter.GaussianBlur(radius=1.2))


def write_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=88)


def generate_samples() -> None:
    random.seed(7)
    for i in range(3):
        write_image(SAMPLES / "fire" / f"sample_fire_{i + 1}.jpg", make_fire_image())
        write_image(SAMPLES / "no_fire" / f"sample_nofire_{i + 1}.jpg", make_nofire_image())
    print(f"Wrote demo samples under {SAMPLES}")


def generate_synthetic_split(n_train: int = 80, n_val: int = 20) -> None:
    random.seed(42)
    for i in range(n_train):
        write_image(IMAGES_TRAIN / "fire" / f"syn_fire_{i:03d}.jpg", make_fire_image())
        write_image(IMAGES_TRAIN / "no_fire" / f"syn_nofire_{i:03d}.jpg", make_nofire_image())
    for i in range(n_val):
        write_image(IMAGES_VAL / "fire" / f"syn_fire_{i:03d}.jpg", make_fire_image())
        write_image(IMAGES_VAL / "no_fire" / f"syn_nofire_{i:03d}.jpg", make_nofire_image())
    print(f"Synthetic ImageFolder data: {n_train} train + {n_val} val per class")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download / generate project datasets.")
    parser.add_argument("--synthetic-images", action="store_true", help="Create synthetic train/val images")
    parser.add_argument("--samples-only", action="store_true", help="Only write bundled Streamlit samples")
    args = parser.parse_args()

    generate_samples()
    if args.samples_only:
        print(IMAGE_DATASET_NOTES)
        return 0

    download_tabular()
    if args.synthetic_images:
        generate_synthetic_split()
    print(IMAGE_DATASET_NOTES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
