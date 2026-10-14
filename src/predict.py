"""Load trained models and run image / weather / fused inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
TABULAR_MODEL = ROOT / "models" / "tabular_rf.joblib"
IMAGE_MODEL = ROOT / "models" / "image_mobilenet.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["no_fire", "fire"]

IMAGE_TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def fuse_probabilities(
    p_image: float | None,
    p_tabular: float | None,
    image_weight: float = 0.6,
) -> dict:
    """Late fusion of P(fire). Either modality may be omitted."""
    if p_image is None and p_tabular is None:
        raise ValueError("Need at least one probability")
    if p_image is None:
        p = float(p_tabular)
        used = "weather"
    elif p_tabular is None:
        p = float(p_image)
        used = "image"
    else:
        w = min(max(float(image_weight), 0.0), 1.0)
        p = w * float(p_image) + (1.0 - w) * float(p_tabular)
        used = "fusion"
    label = "fire" if p >= 0.5 else "no_fire"
    return {
        "p_fire": p,
        "p_no_fire": 1.0 - p,
        "label": label,
        "mode": used,
        "p_image": p_image,
        "p_tabular": p_tabular,
        "image_weight": image_weight,
    }


class TabularPredictor:
    def __init__(self, path: Path = TABULAR_MODEL) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Train first: python src/train_tabular.py"
            )
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]

    def predict_proba(self, values: dict) -> dict:
        row = pd.DataFrame([{k: float(values[k]) for k in self.features}])
        proba = self.pipeline.predict_proba(row)[0]
        classes = list(self.pipeline.classes_)
        p_fire = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
        return {
            "p_fire": p_fire,
            "p_no_fire": 1.0 - p_fire,
            "label": "fire" if p_fire >= 0.5 else "no_fire",
        }


class ImagePredictor:
    def __init__(self, path: Path = IMAGE_MODEL) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Train first: python src/train_image.py"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.class_names = payload.get("class_names", CLASS_NAMES)
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 2)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self.model = model.to(self.device)

    def predict_proba(self, image: Image.Image) -> dict:
        rgb = image.convert("RGB")
        tensor = IMAGE_TF(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
        fire_idx = self.class_names.index("fire") if "fire" in self.class_names else 1
        p_fire = float(proba[fire_idx])
        return {
            "p_fire": p_fire,
            "p_no_fire": 1.0 - p_fire,
            "label": "fire" if p_fire >= 0.5 else "no_fire",
            "raw": {name: float(p) for name, p in zip(self.class_names, proba)},
        }


def default_weather() -> dict:
    return {
        "Temperature": 32.0,
        "RH": 45.0,
        "Ws": 16.0,
        "Rain": 0.0,
        "FFMC": 86.0,
        "DMC": 16.0,
        "DC": 50.0,
        "ISI": 8.0,
        "BUI": 17.0,
        "FWI": 8.0,
    }
