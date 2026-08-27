"""
Train a MobileNetV2-based land cover classifier on EuroSAT (RGB, 10 classes).

Usage:
    python scripts/train_landcover.py [--epochs N] [--batch-size N]
                                       [--data-dir PATH] [--output-path PATH]

Per-epoch metrics are written to reports/train_log_<timestamp>.csv as training
runs, so the log survives regardless of whether stdout is captured.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import EuroSAT
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
NUM_CLASSES = len(CLASSES)


def get_transforms(augment: bool):
    base = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


def build_model() -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Freeze all backbone layers first
    for param in model.features.parameters():
        param.requires_grad = False
    # Unfreeze the last 3 InvertedResidual blocks (indices 15, 16, 17 of features)
    # MobileNetV2 has 19 feature blocks (0–18). Unfreezing the last 3 lets the
    # network adapt high-level features to satellite spectral patterns, which
    # differ significantly from ImageNet natural photos.
    for block in model.features[15:]:
        for param in block.parameters():
            param.requires_grad = True
    # Replace classifier head for 10 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_optimizer(model: nn.Module, head_lr: float = 1e-3, backbone_lr: float = 1e-4):
    """
    Differential learning rates: unfrozen backbone blocks at backbone_lr,
    classifier head at head_lr.  Frozen params are excluded entirely.
    """
    backbone_params = [p for p in model.features[15:].parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    return torch.optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ])


def stratified_split(dataset, train_frac=0.70, val_frac=0.15, seed=42):
    labels = [dataset.targets[i] for i in range(len(dataset))]

    # First split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_frac,
                                   random_state=seed)
    train_idx, rest_idx = next(sss1.split(np.zeros(len(labels)), labels))

    # Second split: val vs test from rest
    rest_labels = [labels[i] for i in rest_idx]
    val_ratio = val_frac / (1 - train_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_ratio,
                                   random_state=seed)
    val_local, test_local = next(
        sss2.split(np.zeros(len(rest_labels)), rest_labels))

    val_idx = rest_idx[val_local]
    test_idx = rest_idx[test_local]
    return train_idx, val_idx, test_idx


def plot_confusion_matrix(cm, classes, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Land Cover Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load full dataset (no transform yet, just for indices) ──────────────
    base_ds = EuroSAT(root=args.data_dir, download=False,
                       transform=get_transforms(augment=False))
    train_idx, val_idx, test_idx = stratified_split(base_ds)

    train_ds = Subset(
        EuroSAT(root=args.data_dir, download=False,
                transform=get_transforms(augment=True)),
        train_idx,
    )
    val_ds = Subset(base_ds, val_idx)
    test_ds = Subset(base_ds, test_idx)

    print(f"Split  →  train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, head_lr=1e-3, backbone_lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    best_val_acc = 0.0
    best_epoch = 0
    patience = 5
    no_improve = 0
    best_state = None

    # ── Persistent per-epoch CSV log ──────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    log_timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("reports", f"train_log_{log_timestamp}.csv")
    log_fields = ["epoch", "train_loss", "train_acc", "val_acc", "elapsed_s", "best_val_acc"]
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()
    log_file.flush()
    print(f"Epoch log → {log_path}")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  elapsed={elapsed:.1f}s"
        )

        # Write row immediately so the log is usable even if training is interrupted
        log_writer.writerow({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_acc": round(val_acc, 6),
            "elapsed_s": round(elapsed, 1),
            "best_val_acc": round(max(best_val_acc, val_acc), 6),
        })
        log_file.flush()

        scheduler.step()

        # ── Early stopping ─────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(best val_acc={best_val_acc:.4f} at epoch {best_epoch})")
                break

    total_time = time.time() - t0
    log_file.close()
    print(f"\nTraining finished in {total_time:.1f}s")
    print(f"Epoch log saved to {log_path}")

    # Restore best weights
    model.load_state_dict(best_state)
    model = model.to(device)

    # ── Test evaluation ────────────────────────────────────────────────────
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest accuracy: {test_acc:.4f}  ({int(test_acc * len(all_labels))}/{len(all_labels)} correct)")

    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join("reports", "landcover_confusion_matrix.png")
    plot_confusion_matrix(cm, CLASSES, cm_path)

    # ── Save model ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(best_state, args.output_path)
    print(f"Model saved to {args.output_path}")

    meta = {
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
        "architecture": "mobilenet_v2",
        "input_size": [3, 64, 64],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "test_accuracy": round(test_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "best_epoch": best_epoch,
        "total_training_seconds": round(total_time, 1),
        "device": str(device),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }
    meta_path = args.output_path.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EuroSAT land cover classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--output-path", type=str,
                        default="models/landcover_classifier.pt")
    args = parser.parse_args()
    train(args)
