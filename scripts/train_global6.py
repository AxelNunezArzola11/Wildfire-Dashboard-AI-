"""
scripts/train_global6.py — Train a MobileNetV2 Global-6 land-cover classifier.

Training data
-------------
  Source 1: EuroSAT RGB JPEG patches (data/eurosat/2750/) — 10 original classes
            remapped to Global-6 via EUROSAT_TO_GLOBAL6.  80% used for training,
            20% held out as eurosat_val_manifest.csv (already written, seed=42).
  Source 2: Angola Sentinel-2 npy patches (data/patches_angola/) — 6 Global-6
            classes.  Train/val split from train_manifest.csv / val_manifest.csv.

Class weighting (source-aware two-factor scheme)
-------------------------------------------------
  Each training sample receives a per-sample weight:
      sample_weight = base_class_weight × source_boost

  base_class_weight = sqrt(N_total / N_class), normalised max=1.0.
    Rationale: equalises gradient mass across the 6 classes in aggregate.

  source_boost = 1.0 for all samples EXCEPT:
    Forest_Vegetation / Angola  → boost = (N_es * 0.25) / (N_ao * 0.75)  ≈ 53×
    Built_up          / Angola  → boost = (N_es * 0.25) / (N_ao * 0.75)  ≈ 63×

    Target: Angola samples contribute ~25% of within-class gradient mass for
    Forest_Vegetation and Built_up. Without boosting, Angola contributes only
    0.6–0.5% (160–188× European dominance), causing the model to learn only
    the European spectral signature and misclassify Angolan woodland/built as Wetland.

  Sampling: WeightedRandomSampler draws N = len(combined_ds) samples per epoch
    with replacement proportional to per-sample weight.  This changes *how often*
    each sample is seen, not just how hard the gradient is.

  Angola FV/Built_up samples are drawn ~40–47× per epoch on average.
  Stronger augmentation (flips, rotation, colour jitter, blur) is applied to
  ALL Angola samples to reduce memorisation risk from high-repetition sampling.

Validation
----------
  Two separate val sets evaluated every epoch:
    • EuroSAT val  (5,400 samples, 4 classes Forest/Crop/Water/Built_up)
      → regression guard: did retraining hurt European accuracy?
    • Angola val   (118 samples, 6 classes)
      → generalisation check: can the model classify tropical land covers?
  Per-class accuracy is logged for BOTH — never blended.

Usage
-----
    python3 scripts/train_global6.py [--epochs N] [--batch-size N]
                                     [--lr-head F] [--lr-backbone F]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

# project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from landcover_schema import (
    EUROSAT_TO_GLOBAL6,
    GLOBAL6_CLASSES,
    GLOBAL6_LABEL,
)

NUM_CLASSES = len(GLOBAL6_CLASSES)
SEED = 42

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transform(augment: bool) -> transforms.Compose:
    base = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    if augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


def get_angola_transform(augment: bool) -> transforms.Compose:
    """
    Stronger augmentation for Angola patches specifically.
    Applied to ALL Angola training samples because the WRS repeats each
    image ~10–47× per epoch; richer augmentation is the primary defence
    against memorisation.
    Extra ops vs. EuroSAT: larger rotation, added GaussianBlur, wider jitter.
    """
    base = [
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    if augment:
        # These ops work on float tensors; apply before Normalize
        return _AngolaAugment()
    return transforms.Compose(base)


_NORMALIZE = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

# Angola augmentation pipeline operates on float32 tensors (CHW, [0,1])
# before the ImageNet normalisation step.
class _AngolaAugment:
    """Callable that applies strong augmentation to a (3,H,W) float32 tensor."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        # Horizontal flip
        if torch.rand(1).item() > 0.5:
            t = torch.flip(t, [2])
        # Vertical flip
        if torch.rand(1).item() > 0.5:
            t = torch.flip(t, [1])
        # 90° rotation (random k ∈ {0,1,2,3})
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            t = torch.rot90(t, k, [1, 2])
        # Brightness jitter ±25%
        factor = 0.75 + torch.rand(1).item() * 0.5   # [0.75, 1.25]
        t = torch.clamp(t * factor, 0.0, 1.0)
        # Contrast jitter: stretch each channel around its mean
        c_factor = 0.75 + torch.rand(1).item() * 0.5
        mean = t.mean(dim=[1, 2], keepdim=True)
        t = torch.clamp(mean + (t - mean) * c_factor, 0.0, 1.0)
        # Gaussian blur (simple 3×3 box blur as cheap approximation)
        if torch.rand(1).item() > 0.5:
            # unfold → mean pooling approximation
            pad = torch.nn.functional.pad(t.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
            t = torch.nn.functional.avg_pool2d(pad, 3, stride=1, padding=0).squeeze(0)
        return _NORMALIZE(t)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class EuroSATGlobal6Dataset(Dataset):
    """
    EuroSAT JPEG patches remapped to Global-6 labels.
    Reads from data/eurosat/2750/{OrigClass}/*.jpg.
    """

    def __init__(self, records: list[dict], transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r   = self.records[idx]
        img = Image.open(r["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(r["global6_index"])
        return img, label


class AngolaPatches(Dataset):
    """
    Angola Sentinel-2 npy patches — shape (3, 64, 64), float32, reflectance [0,1].
    Converts to the same normalised tensor space as EuroSAT.
    augment=True applies the stronger _AngolaAugment pipeline.
    """

    def __init__(self, records: list[dict], augment: bool = False):
        self.records  = records
        self.augment  = augment
        self._aug_fn  = _AngolaAugment() if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r   = self.records[idx]
        arr = np.load(r["filepath"]).astype(np.float32)   # (3, 64, 64)
        arr = np.clip(arr, 0.0, 1.0)
        t   = torch.from_numpy(arr)                        # (3, 64, 64)

        if self._aug_fn is not None:
            t = self._aug_fn(t)
        else:
            t = _NORMALIZE(t)
        label = int(r["global6_index"])
        return t, label


class CombinedDataset(Dataset):
    """
    Concatenates EuroSAT and Angola datasets.
    Also stores per-sample weights for WeightedRandomSampler.
    """

    def __init__(self, eurosat_ds: Dataset, angola_ds: Dataset,
                 sample_weights: list[float]):
        self.datasets       = [eurosat_ds, angola_ds]
        self.lengths        = [len(eurosat_ds), len(angola_ds)]
        self.total          = sum(self.lengths)
        self.sample_weights = sample_weights  # one weight per index

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        if idx < self.lengths[0]:
            return self.datasets[0][idx]
        return self.datasets[1][idx - self.lengths[0]]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_eurosat_records(data_dir: Path, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """
    Load all EuroSAT global6_labels.csv, apply the same 80/20 stratified split
    as train_landcover.py (seed=42), return (train_records, val_records).
    """
    records = []
    with open(data_dir / "eurosat" / "global6_labels.csv") as f:
        for row in csv.DictReader(f):
            records.append(row)

    labels  = [int(r["global6_index"]) for r in records]
    indices = np.arange(len(records))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(sss.split(indices, labels))
    return [records[i] for i in train_idx], [records[i] for i in val_idx]


def load_angola_records(patches_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load Angola train and val manifests."""
    def _read(path):
        with open(path) as f:
            return list(csv.DictReader(f))
    return _read(patches_dir / "train_manifest.csv"), _read(patches_dir / "val_manifest.csv")


# ---------------------------------------------------------------------------
# Class weights (sqrt-inverse-frequency)
# ---------------------------------------------------------------------------

# Angola classes that receive a source boost so their samples represent
# ~25% of within-class gradient mass (vs. 0.5–0.6% without boosting).
_ANGOLA_BOOST_CLASSES = {"Forest_Vegetation", "Built_up"}
_ANGOLA_BOOST_TARGET  = 0.25   # fraction of within-class loss mass to assign to Angola


def compute_sample_weights(
    train_records_es: list[dict],
    train_records_ao: list[dict],
) -> tuple[list[float], torch.Tensor]:
    """
    Returns (sample_weights_list, class_weight_tensor).

    sample_weights_list: one float per training sample (EuroSAT first, then Angola),
        used to construct a WeightedRandomSampler.

    class_weight_tensor: passed to CrossEntropyLoss(weight=...) for the CE term;
        uses the sqrt-inverse-frequency base weights so the loss scale is
        consistent with the sampling distribution.

    Two-factor weight per sample:
        w = base_class_weight(y) × source_boost(y, source)

    source_boost is 1.0 everywhere except:
        Forest_Vegetation/Angola and Built_up/Angola, where it is set so
        Angola samples contribute _ANGOLA_BOOST_TARGET of the within-class mass.
        boost = (N_es * target) / (N_ao * (1 - target))
    """
    es_by_class: dict[str, int] = defaultdict(int)
    for r in train_records_es:
        es_by_class[r["global6_class"]] += 1
    ao_by_class: dict[str, int] = defaultdict(int)
    for r in train_records_ao:
        ao_by_class[r["global6_class"]] += 1

    counts = {cls: es_by_class[cls] + ao_by_class[cls] for cls in GLOBAL6_CLASSES}
    total  = sum(counts.values())

    # Base class weight (sqrt-inverse-frequency, normalised max=1)
    base_w_raw = {cls: math.sqrt(total / max(counts[cls], 1)) for cls in GLOBAL6_CLASSES}
    max_bw     = max(base_w_raw.values())
    base_w     = {cls: base_w_raw[cls] / max_bw for cls in GLOBAL6_CLASSES}

    # Source boost for FV and Built_up Angola samples
    def _boost(cls: str) -> float:
        n_es = es_by_class[cls]
        n_ao = ao_by_class[cls]
        if cls not in _ANGOLA_BOOST_CLASSES or n_ao == 0:
            return 1.0
        return (n_es * _ANGOLA_BOOST_TARGET) / (n_ao * (1.0 - _ANGOLA_BOOST_TARGET))

    boost = {cls: _boost(cls) for cls in GLOBAL6_CLASSES}

    # Print table
    print("\n  Source-aware per-sample weights (base × source_boost):")
    print(f"  {'Class':<24}  {'Source':<8}  {'N':>6}  {'base_w':>8}  {'boost':>8}  {'sample_w':>10}")
    print("  " + "-" * 70)
    for cls in GLOBAL6_CLASSES:
        for src, n in [("Europe", es_by_class[cls]), ("Angola", ao_by_class[cls])]:
            if n == 0:
                continue
            b   = boost[cls] if src == "Angola" and cls in _ANGOLA_BOOST_CLASSES else 1.0
            sw  = base_w[cls] * b
            print(f"  {cls:<24}  {src:<8}  {n:>6,}  {base_w[cls]:>8.4f}  {b:>8.2f}  {sw:>10.4f}")

    # Build per-sample weight list: EuroSAT records first, then Angola
    sample_weights: list[float] = []
    for r in train_records_es:
        sample_weights.append(base_w[r["global6_class"]])   # source_boost=1 for EuroSAT
    for r in train_records_ao:
        cls = r["global6_class"]
        b   = boost[cls] if cls in _ANGOLA_BOOST_CLASSES else 1.0
        sample_weights.append(base_w[cls] * b)

    # Class weight tensor for CrossEntropyLoss
    class_w_tensor = torch.tensor([base_w[cls] for cls in GLOBAL6_CLASSES],
                                   dtype=torch.float32)

    # Expected draws diagnostic
    sw_tensor = torch.tensor(sample_weights)
    total_sw  = sw_tensor.sum().item()
    n_epoch   = len(sample_weights)
    print(f"\n  WeightedRandomSampler — expected draws per epoch (N={n_epoch:,}):")
    print(f"  {'Class':<24}  {'ES_draws':>10}  {'AO_draws':>10}  {'AO_frac':>8}")
    for cls in GLOBAL6_CLASSES:
        es_idx = [i for i, r in enumerate(train_records_es) if r["global6_class"] == cls]
        ao_idx = [len(train_records_es) + i
                  for i, r in enumerate(train_records_ao) if r["global6_class"] == cls]
        es_d = n_epoch * sum(sample_weights[i] for i in es_idx) / total_sw if es_idx else 0.0
        ao_d = n_epoch * sum(sample_weights[i] for i in ao_idx) / total_sw if ao_idx else 0.0
        frac = ao_d / (es_d + ao_d) if (es_d + ao_d) > 0 else 0.0
        print(f"  {cls:<24}  {es_d:>10.1f}  {ao_d:>10.1f}  {frac:>8.3f}")

    return sample_weights, class_w_tensor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    # Unfreeze last 3 InvertedResidual blocks (same as Fase A baseline)
    for block in model.features[15:]:
        for param in block.parameters():
            param.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_optimizer(model: nn.Module, head_lr: float, backbone_lr: float):
    backbone_params = [p for p in model.features[15:].parameters() if p.requires_grad]
    head_params     = list(model.classifier.parameters())
    return torch.optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ])


# ---------------------------------------------------------------------------
# Per-class accuracy helper
# ---------------------------------------------------------------------------

def per_class_accuracy(all_preds: list[int], all_labels: list[int],
                        classes: list[str]) -> dict[str, float]:
    correct = defaultdict(int)
    total   = defaultdict(int)
    for p, t in zip(all_preds, all_labels):
        total[t]   += 1
        correct[t] += int(p == t)
    accs = {}
    for i, cls in enumerate(classes):
        if total[i] > 0:
            accs[cls] = round(correct[i] / total[i], 4)
        else:
            accs[cls] = None   # class absent from this val set
    return accs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    data_dir    = Path(args.data_dir)
    patches_dir = data_dir / "patches_angola"
    os.makedirs("reports", exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading datasets…")
    es_train_rec, es_val_rec = load_eurosat_records(data_dir)
    ao_train_rec, ao_val_rec = load_angola_records(patches_dir)

    print(f"  EuroSAT  train={len(es_train_rec):,}  val={len(es_val_rec):,}")
    print(f"  Angola   train={len(ao_train_rec):,}   val={len(ao_val_rec):,}")
    print(f"  Combined train total: {len(es_train_rec) + len(ao_train_rec):,}")

    sample_weights, class_weights = compute_sample_weights(es_train_rec, ao_train_rec)

    # Datasets
    aug_transform  = get_transform(augment=True)
    eval_transform = get_transform(augment=False)

    es_train_ds = EuroSATGlobal6Dataset(es_train_rec, transform=aug_transform)
    ao_train_ds = AngolaPatches(ao_train_rec, augment=True)  # strong augmentation
    train_ds    = CombinedDataset(es_train_ds, ao_train_ds, sample_weights)

    es_val_ds   = EuroSATGlobal6Dataset(es_val_rec, transform=eval_transform)
    ao_val_ds   = AngolaPatches(ao_val_rec, augment=False)

    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float64),
        num_samples = len(train_ds),
        replacement = True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=2, pin_memory=False)
    es_val_loader = DataLoader(es_val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=2, pin_memory=False)
    ao_val_loader = DataLoader(ao_val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=1, pin_memory=False)

    # ── Model ────────────────────────────────────────────────────────────────
    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = build_optimizer(model, head_lr=args.lr_head, backbone_lr=args.lr_backbone)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # ── Logging ──────────────────────────────────────────────────────────────
    ts       = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"reports/train_global6_log_{ts}.csv"
    log_fields = [
        "epoch", "train_loss", "train_acc",
        "es_val_acc",        # EuroSAT val overall (4-class subset)
        "ao_val_acc",        # Angola val overall (6-class)
        # per-class EuroSAT val
        "es_Forest_Vegetation", "es_Cropland", "es_Water", "es_Built_up",
        # per-class Angola val (all 6)
        "ao_Forest_Vegetation", "ao_Cropland", "ao_Water",
        "ao_Built_up", "ao_Bare_Sparse", "ao_Wetland",
        "elapsed_s", "best_es_val_acc",
    ]
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()
    log_file.flush()
    print(f"\nEpoch log → {log_path}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_es_val_acc = 0.0
    best_state      = None
    best_epoch      = 0
    patience        = 7
    no_improve      = 0
    t0              = time.time()

    header = (
        f"\n{'Ep':>3}  {'loss':>7}  {'tr_acc':>7}  "
        f"{'ES_val':>7}  {'AO_val':>7}  "
        f"{'ES Fv':>7} {'ES Cr':>7} {'ES Wa':>7} {'ES Bu':>7}  "
        f"{'AO Fv':>7} {'AO Cr':>7} {'AO Wa':>7} {'AO Bu':>7} {'AO Ba':>7} {'AO We':>7}"
    )
    print(header)
    print("-" * len(header.rstrip()))

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        running_loss = correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct      += (out.argmax(1) == labels).sum().item()
            total        += imgs.size(0)
        train_loss = running_loss / total
        train_acc  = correct / total

        # EuroSAT val
        model.eval()
        es_preds, es_labels = [], []
        with torch.no_grad():
            for imgs, labels in es_val_loader:
                out = model(imgs.to(device))
                es_preds.extend(out.argmax(1).cpu().tolist())
                es_labels.extend(labels.tolist())
        es_overall = sum(p == l for p, l in zip(es_preds, es_labels)) / len(es_labels)
        es_pc = per_class_accuracy(es_preds, es_labels, GLOBAL6_CLASSES)

        # Angola val
        ao_preds, ao_labels = [], []
        with torch.no_grad():
            for imgs, labels in ao_val_loader:
                out = model(imgs.to(device))
                ao_preds.extend(out.argmax(1).cpu().tolist())
                ao_labels.extend(labels.tolist())
        ao_overall = sum(p == l for p, l in zip(ao_preds, ao_labels)) / len(ao_labels)
        ao_pc = per_class_accuracy(ao_preds, ao_labels, GLOBAL6_CLASSES)

        elapsed = time.time() - t0
        scheduler.step()

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "  n/a "

        print(
            f"{epoch:>3}  {train_loss:>7.4f}  {train_acc:>7.4f}  "
            f"{es_overall:>7.4f}  {ao_overall:>7.4f}  "
            f"{_fmt(es_pc['Forest_Vegetation']):>7} "
            f"{_fmt(es_pc['Cropland']):>7} "
            f"{_fmt(es_pc['Water']):>7} "
            f"{_fmt(es_pc['Built_up']):>7}  "
            f"{_fmt(ao_pc['Forest_Vegetation']):>7} "
            f"{_fmt(ao_pc['Cropland']):>7} "
            f"{_fmt(ao_pc['Water']):>7} "
            f"{_fmt(ao_pc['Built_up']):>7} "
            f"{_fmt(ao_pc['Bare_Sparse']):>7} "
            f"{_fmt(ao_pc['Wetland']):>7}"
        )

        log_writer.writerow({
            "epoch":            epoch,
            "train_loss":       round(train_loss, 6),
            "train_acc":        round(train_acc,  6),
            "es_val_acc":       round(es_overall, 6),
            "ao_val_acc":       round(ao_overall, 6),
            "es_Forest_Vegetation": es_pc["Forest_Vegetation"],
            "es_Cropland":          es_pc["Cropland"],
            "es_Water":             es_pc["Water"],
            "es_Built_up":          es_pc["Built_up"],
            "ao_Forest_Vegetation": ao_pc["Forest_Vegetation"],
            "ao_Cropland":          ao_pc["Cropland"],
            "ao_Water":             ao_pc["Water"],
            "ao_Built_up":          ao_pc["Built_up"],
            "ao_Bare_Sparse":       ao_pc["Bare_Sparse"],
            "ao_Wetland":           ao_pc["Wetland"],
            "elapsed_s":        round(elapsed, 1),
            "best_es_val_acc":  round(max(best_es_val_acc, es_overall), 6),
        })
        log_file.flush()

        # Early stopping — keyed on EuroSAT val (the stable signal; Angola val
        # is noisier at 118 samples and would cause premature stops)
        if es_overall > best_es_val_acc:
            best_es_val_acc = es_overall
            best_epoch      = epoch
            no_improve      = 0
            best_state      = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best ES val={best_es_val_acc:.4f} @ epoch {best_epoch})")
                break

    total_time = time.time() - t0
    log_file.close()
    print(f"\nTraining finished in {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Final evaluation with best weights ────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(device).eval()

    def _eval_loader(loader, name, n_samples):
        preds, labels_all = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                out = model(imgs.to(device))
                preds.extend(out.argmax(1).cpu().tolist())
                labels_all.extend(labels.tolist())
        overall = sum(p == l for p, l in zip(preds, labels_all)) / max(len(labels_all), 1)
        pc = per_class_accuracy(preds, labels_all, GLOBAL6_CLASSES)
        print(f"\n  {name} final accuracy (best epoch {best_epoch}):")
        print(f"  Overall: {overall:.4f}  ({int(overall*n_samples)}/{n_samples})")
        print(f"  {'Class':<24}  {'Acc':>7}  {'Note'}")
        print("  " + "-"*48)
        for cls in GLOBAL6_CLASSES:
            v = pc[cls]
            note = "absent from this val set" if v is None else ""
            print(f"  {cls:<24}  {v if v is not None else '  n/a ':>7}  {note}")
        return overall, pc

    print(f"\n{'='*60}")
    print(f"  Final Results (best weights, epoch {best_epoch})")
    print(f"{'='*60}")
    es_final_acc, es_final_pc = _eval_loader(es_val_loader, "EuroSAT val", len(es_val_rec))
    ao_final_acc, ao_final_pc = _eval_loader(ao_val_loader, "Angola val",  len(ao_val_rec))

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(best_state, args.output_path)
    print(f"\nModel saved: {args.output_path}")

    meta = {
        "classes":              GLOBAL6_CLASSES,
        "num_classes":          NUM_CLASSES,
        "architecture":         "mobilenet_v2",
        "input_size":           [3, 64, 64],
        "normalize_mean":       _IMAGENET_MEAN,
        "normalize_std":        _IMAGENET_STD,
        "weighting_scheme":     "source_aware_sqrt_inv_freq_wrs",
        "angola_boost_target":  _ANGOLA_BOOST_TARGET,
        "angola_boost_classes": sorted(_ANGOLA_BOOST_CLASSES),
        "class_weights":        {cls: round(w, 4) for cls, w in
                                  zip(GLOBAL6_CLASSES,
                                      compute_sample_weights(es_train_rec, ao_train_rec)[1].tolist())},
        "best_epoch":           best_epoch,
        "total_training_s":     round(total_time, 1),
        "device":               str(device),
        "train_size_eurosat":   len(es_train_rec),
        "train_size_angola":    len(ao_train_rec),
        "val_size_eurosat":     len(es_val_rec),
        "val_size_angola":      len(ao_val_rec),
        "es_val_overall_acc":   round(es_final_acc, 4),
        "ao_val_overall_acc":   round(ao_final_acc, 4),
        "es_val_per_class":     {k: (round(v, 4) if v is not None else None)
                                  for k, v in es_final_pc.items()},
        "ao_val_per_class":     {k: (round(v, 4) if v is not None else None)
                                  for k, v in ao_final_pc.items()},
        "epoch_log":            log_path,
    }
    meta_path = args.output_path.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Global-6 land cover classifier")
    parser.add_argument("--epochs",       type=int,   default=15)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr-head",      type=float, default=1e-3)
    parser.add_argument("--lr-backbone",  type=float, default=1e-4)
    parser.add_argument("--data-dir",     type=str,   default="data/")
    parser.add_argument("--output-path",  type=str,
                        default="models/global6_classifier.pt")
    args = parser.parse_args()
    train(args)
