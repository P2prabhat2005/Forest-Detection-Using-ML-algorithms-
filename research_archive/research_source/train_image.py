"""Train MobileNetV2 (transfer learning) on fire vs no_fire ImageFolder data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "images" / "train"
VAL_DIR = ROOT / "data" / "images" / "val"
MODEL_PATH = ROOT / "models" / "image_mobilenet.pt"
METRICS_PATH = ROOT / "models" / "image_metrics.json"

CLASS_NAMES = ["no_fire", "fire"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def require_imagefolder(path: Path) -> None:
    fire = path / "fire"
    no_fire = path / "no_fire"
    if not fire.is_dir() or not no_fire.is_dir():
        raise FileNotFoundError(
            "Image training data not found. Expected:\n"
            f"  {path / 'fire'}\n"
            f"  {path / 'no_fire'}\n\n"
            "Unzip a fire-image dataset into that layout, or generate a local set:\n"
            "  python scripts/download_data.py --synthetic-images\n"
        )
    if not any(fire.iterdir()) or not any(no_fire.iterdir()):
        raise FileNotFoundError(
            f"{path} class folders are empty. Add images or run "
            "python scripts/download_data.py --synthetic-images"
        )


def build_model(device: torch.device) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model.to(device)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += accuracy(logits.detach(), y) * bs
        n += bs
    return total_loss / n, total_acc / n


def remap_targets(dataset: datasets.ImageFolder) -> None:
    """Ensure class index 0 = no_fire, 1 = fire regardless of folder sort."""
    name_to_idx = dataset.class_to_idx
    if set(name_to_idx) != set(CLASS_NAMES):
        raise ValueError(
            f"Expected folders {CLASS_NAMES}, got {sorted(name_to_idx)}"
        )
    mapping = {name_to_idx["no_fire"]: 0, name_to_idx["fire"]: 1}
    dataset.samples = [(p, mapping[y]) for p, y in dataset.samples]
    dataset.targets = [mapping[y] for y in dataset.targets]
    dataset.class_to_idx = {"no_fire": 0, "fire": 1}
    dataset.classes = CLASS_NAMES


def train(epochs_frozen: int = 3, epochs_ft: int = 2, batch_size: int = 16) -> None:
    require_imagefolder(TRAIN_DIR)
    require_imagefolder(VAL_DIR)

    train_tf, eval_tf = build_transforms()
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tf)
    remap_targets(train_ds)
    remap_targets(val_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    history: list[dict] = []
    best_val = -1.0
    best_state = None

    def loop(n_epochs: int, tag: str) -> None:
        nonlocal best_val, best_state
        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)
            history.append(
                {
                    "phase": tag,
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                }
            )
            print(
                f"[{tag} {epoch}/{n_epochs}] "
                f"train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_loss={va_loss:.3f}"
            )
            if va_acc >= best_val:
                best_val = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    loop(epochs_frozen, "frozen")

    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loop(epochs_ft, "finetune")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "class_names": CLASS_NAMES,
        "val_accuracy": best_val,
    }
    torch.save(payload, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "val_accuracy": best_val,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "class_names": CLASS_NAMES,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model -> {MODEL_PATH} (best val_acc={best_val:.3f})")


if __name__ == "__main__":
    try:
        train()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
