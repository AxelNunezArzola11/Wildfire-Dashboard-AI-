"""
scripts/calibrate_temperature.py — Temperature scaling for land cover classifiers.

Fits a single scalar temperature T per model by minimising NLL on the held-out
validation logits.  Writes T values to models/*_temp.json and reports ECE
before/after.

Usage:
    python3 scripts/calibrate_temperature.py

Output files (created/overwritten):
    models/global6_temperature.json    {"T": 1.234, "source": "..."}
    models/eurosat10_temperature.json  {"T": 1.234, "source": "..."}

ECE calculation follows the 15-bin equal-width scheme standard in calibration
literature (Guo et al. 2017 "On Calibration of Modern Neural Networks").
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_NORMALIZE     = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

GLOBAL6_CLASSES = [
    "Forest_Vegetation", "Cropland", "Water", "Built_up", "Bare_Sparse", "Wetland"
]
EUROSAT10_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# ── Model loader ─────────────────────────────────────────────────────────────

def _load_model(ckpt_path: str, num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Preprocessing pipelines (must match training) ────────────────────────────

# EuroSAT10 val: images are JPEG uint8 full-range (already in DN [0,255])
# Global-6 val EuroSAT images: same JPEG pipeline
_EUROSAT_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    _NORMALIZE,
])

# Angola val: npy patches shape (3,64,64), float32 reflectance [0,~0.25]
# Training normalised directly: clip [0,1] → _NORMALIZE (no stretch)
def _angola_to_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    t = torch.from_numpy(arr)   # (3,64,64)
    return _NORMALIZE(t)


# ── Collect logits ────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_eurosat_logits(
    model: nn.Module,
    manifest_path: str,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (logits [N, C], labels [N]) for the EuroSAT val manifest.
    Uses the same JPEG → Resize(64) → ToTensor → Normalize pipeline as training.
    """
    all_logits, all_labels = [], []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_samples:
        rows = rows[:max_samples]

    for row in rows:
        img_path  = str(ROOT / row["filepath"])
        label_idx = int(row["global6_index"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"  [WARN] skip {img_path}: {exc}")
            continue
        tensor = _EUROSAT_TRANSFORM(img).unsqueeze(0)   # (1,3,64,64)
        logits = model(tensor).squeeze(0)                # (C,)
        all_logits.append(logits.numpy())
        all_labels.append(label_idx)

    return np.array(all_logits, dtype=np.float32), np.array(all_labels, dtype=np.int64)


@torch.no_grad()
def collect_angola_logits(
    model: nn.Module,
    manifest_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (logits [N, C], labels [N]) for the Angola val manifest.
    Uses the same clip → _NORMALIZE pipeline as training (no p2/p98 stretch).
    """
    all_logits, all_labels = [], []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    for row in rows:
        npy_path  = str(ROOT / row["filepath"])
        label_idx = int(row["global6_index"])
        try:
            arr = np.load(npy_path)
        except Exception as exc:
            print(f"  [WARN] skip {npy_path}: {exc}")
            continue
        tensor = _angola_to_tensor(arr).unsqueeze(0)   # (1,3,64,64)
        logits = model(tensor).squeeze(0)               # (C,)
        all_logits.append(logits.numpy())
        all_labels.append(label_idx)

    return np.array(all_logits, dtype=np.float32), np.array(all_labels, dtype=np.int64)


# ── ECE calculation ───────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (equal-width bins, Guo et al. 2017).

    probs  : (N, C) softmax probabilities
    labels : (N,)  integer class indices
    returns: ECE scalar in [0, 1]
    """
    confidences  = probs.max(axis=1)
    predictions  = probs.argmax(axis=1)
    accuracies   = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = accuracies[mask].mean()
        ece     += mask.sum() / n * abs(avg_conf - avg_acc)
    return float(ece)


# ── Temperature fitting ───────────────────────────────────────────────────────

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Fit a single scalar temperature T via NLL minimisation (LBFGS).

    T > 1 → softens probabilities (reduces overconfidence).
    T < 1 → sharpens probabilities (rarely happens with over-confident models).

    Returns T as a Python float.
    """
    logits_t = torch.from_numpy(logits)     # (N, C)
    labels_t = torch.from_numpy(labels).long()

    log_T = nn.Parameter(torch.zeros(1))   # we optimise log(T) to keep T > 0

    optimizer = torch.optim.LBFGS([log_T], max_iter=500, tolerance_change=1e-9)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        T_val  = torch.exp(log_T)
        scaled = logits_t / T_val
        loss   = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(log_T).item())
    return T


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Temperature scaling calibration")
    print("=" * 70)

    models_dir   = ROOT / "models"
    global6_ckpt  = str(models_dir / "global6_classifier.pt")
    eurosat10_ckpt = str(models_dir / "landcover_classifier.pt")

    es_val_manifest = str(ROOT / "data" / "eurosat" / "eurosat_val_manifest.csv")
    ao_val_manifest = str(ROOT / "data" / "patches_angola" / "val_manifest.csv")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    for p, label in [
        (global6_ckpt,  "global6 checkpoint"),
        (eurosat10_ckpt, "eurosat10 checkpoint"),
        (es_val_manifest, "EuroSAT val manifest"),
        (ao_val_manifest, "Angola val manifest"),
    ]:
        if not os.path.isfile(p):
            print(f"  [ERROR] Missing file: {p} ({label})")
            sys.exit(1)

    # ════════════════════════════════════════════════════════════════
    # GLOBAL-6 MODEL
    # ════════════════════════════════════════════════════════════════
    print("\n── Global-6 model ──────────────────────────────────────────────")
    g6_model = _load_model(global6_ckpt, num_classes=6)

    # EuroSAT val (5400 samples)
    print("  Collecting EuroSAT val logits (5400 samples)...", flush=True)
    g6_es_logits, g6_es_labels = collect_eurosat_logits(
        g6_model, es_val_manifest
    )
    print(f"  → {len(g6_es_labels)} samples loaded")

    # Angola val (118 samples)
    print("  Collecting Angola val logits (118 samples)...", flush=True)
    g6_ao_logits, g6_ao_labels = collect_angola_logits(
        g6_model, ao_val_manifest
    )
    print(f"  → {len(g6_ao_labels)} samples loaded")

    # Combined logits for fitting T (both val sets)
    combined_g6_logits = np.concatenate([g6_es_logits, g6_ao_logits], axis=0)
    combined_g6_labels = np.concatenate([g6_es_labels, g6_ao_labels], axis=0)

    # ECE before calibration
    g6_probs_raw_es = torch.softmax(torch.from_numpy(g6_es_logits), dim=1).numpy()
    g6_probs_raw_ao = torch.softmax(torch.from_numpy(g6_ao_logits), dim=1).numpy()
    ece_g6_es_before = compute_ece(g6_probs_raw_es, g6_es_labels)
    ece_g6_ao_before = compute_ece(g6_probs_raw_ao, g6_ao_labels)
    print(f"\n  ECE BEFORE calibration:")
    print(f"    EuroSAT val  : {ece_g6_es_before:.4f}")
    print(f"    Angola val   : {ece_g6_ao_before:.4f}")

    # Fit T on combined (EuroSAT + Angola) val logits
    print("\n  Fitting T on combined val set (EuroSAT + Angola)...", flush=True)
    T_g6 = fit_temperature(combined_g6_logits, combined_g6_labels)
    print(f"  → Fitted T = {T_g6:.6f}")

    # ECE after calibration
    g6_probs_cal_es = torch.softmax(
        torch.from_numpy(g6_es_logits) / T_g6, dim=1
    ).numpy()
    g6_probs_cal_ao = torch.softmax(
        torch.from_numpy(g6_ao_logits) / T_g6, dim=1
    ).numpy()
    ece_g6_es_after = compute_ece(g6_probs_cal_es, g6_es_labels)
    ece_g6_ao_after = compute_ece(g6_probs_cal_ao, g6_ao_labels)
    print(f"\n  ECE AFTER calibration (T={T_g6:.4f}):")
    print(f"    EuroSAT val  : {ece_g6_es_after:.4f}  (was {ece_g6_es_before:.4f})")
    print(f"    Angola val   : {ece_g6_ao_after:.4f}  (was {ece_g6_ao_before:.4f})")

    # Show confidence shift on known bad patches
    print("\n  Known bad cases — confidence before/after T:")
    _show_known_bad_cases_g6(g6_model, T_g6)

    # Save T
    g6_temp_path = str(models_dir / "global6_temperature.json")
    with open(g6_temp_path, "w") as _tf:
        json.dump(
            {
                "T": T_g6,
                "fit_on": "EuroSAT val (5400) + Angola val (118) — combined NLL minimisation (LBFGS)",
                "ece_eurosat_val_before": round(ece_g6_es_before, 4),
                "ece_eurosat_val_after":  round(ece_g6_es_after, 4),
                "ece_angola_val_before":  round(ece_g6_ao_before, 4),
                "ece_angola_val_after":   round(ece_g6_ao_after, 4),
            },
            _tf,
            indent=2,
        )
    print(f"\n  Saved → {g6_temp_path}")

    # ════════════════════════════════════════════════════════════════
    # EUROSAT-10 MODEL
    # ════════════════════════════════════════════════════════════════
    print("\n── EuroSAT-10 model ────────────────────────────────────────────")
    e10_model = _load_model(eurosat10_ckpt, num_classes=10)

    # EuroSAT-10 val manifest uses global6_index; we need the ORIGINAL 10-class
    # labels.  The eurosat_val_manifest.csv has both original_class and global6_class.
    # For ECE we use the global6_index because we only have a 10-class model here —
    # but the ORIGINAL labels for the 10-class model are the 10-class indices.
    # We re-derive those from original_class:
    _ES10_IDX = {c: i for i, c in enumerate(EUROSAT10_CLASSES)}

    print("  Collecting EuroSAT val logits for eurosat10 (5400 samples)...", flush=True)
    all_logits_e10, all_labels_e10 = [], []
    with open(es_val_manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = str(ROOT / row["filepath"])
            orig_cls = row["original_class"]
            label_e10 = _ES10_IDX.get(orig_cls)
            if label_e10 is None:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as exc:
                print(f"  [WARN] skip {img_path}: {exc}")
                continue
            tensor  = _EUROSAT_TRANSFORM(img).unsqueeze(0)
            logits  = e10_model(tensor).squeeze(0)
            all_logits_e10.append(logits.detach().numpy())
            all_labels_e10.append(label_e10)

    e10_logits = np.array(all_logits_e10, dtype=np.float32)
    e10_labels = np.array(all_labels_e10, dtype=np.int64)
    print(f"  → {len(e10_labels)} samples loaded")

    e10_probs_raw = torch.softmax(torch.from_numpy(e10_logits), dim=1).numpy()
    ece_e10_before = compute_ece(e10_probs_raw, e10_labels)
    print(f"\n  ECE BEFORE calibration:")
    print(f"    EuroSAT val  : {ece_e10_before:.4f}")

    print("\n  Fitting T for eurosat10...", flush=True)
    T_e10 = fit_temperature(e10_logits, e10_labels)
    print(f"  → Fitted T = {T_e10:.6f}")

    e10_probs_cal = torch.softmax(
        torch.from_numpy(e10_logits) / T_e10, dim=1
    ).numpy()
    ece_e10_after = compute_ece(e10_probs_cal, e10_labels)
    print(f"\n  ECE AFTER calibration (T={T_e10:.4f}):")
    print(f"    EuroSAT val  : {ece_e10_after:.4f}  (was {ece_e10_before:.4f})")

    # Save T
    e10_temp_path = str(models_dir / "eurosat10_temperature.json")
    with open(e10_temp_path, "w") as _tf:
        json.dump(
            {
                "T": T_e10,
                "fit_on": "EuroSAT val (5400) — NLL minimisation (LBFGS)",
                "ece_eurosat_val_before": round(ece_e10_before, 4),
                "ece_eurosat_val_after":  round(ece_e10_after, 4),
            },
            _tf,
            indent=2,
        )
    print(f"\n  Saved → {e10_temp_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  global6  T = {T_g6:.4f}")
    print(f"    EuroSAT val  ECE: {ece_g6_es_before:.4f} → {ece_g6_es_after:.4f}  "
          f"({'↓' if ece_g6_es_after < ece_g6_es_before else '↑'}"
          f" {abs(ece_g6_es_after - ece_g6_es_before):.4f})")
    print(f"    Angola val   ECE: {ece_g6_ao_before:.4f} → {ece_g6_ao_after:.4f}  "
          f"({'↓' if ece_g6_ao_after < ece_g6_ao_before else '↑'}"
          f" {abs(ece_g6_ao_after - ece_g6_ao_before):.4f})")
    print(f"  eurosat10 T = {T_e10:.4f}")
    print(f"    EuroSAT val  ECE: {ece_e10_before:.4f} → {ece_e10_after:.4f}  "
          f"({'↓' if ece_e10_after < ece_e10_before else '↑'}"
          f" {abs(ece_e10_after - ece_e10_before):.4f})")


def _show_known_bad_cases_g6(model: nn.Module, T: float) -> None:
    """
    Re-run confidence check on the specific bad-case patches documented in
    JUDGE.md.  This shows the confidence BEFORE and AFTER temperature scaling.
    The prediction does NOT change — T only shifts the probability magnitude.
    """
    # Known bad patches from JUDGE.md Phase B section
    bad_patches = [
        ("data/patches_angola/Forest_Vegetation/patch_m4.46762_15.67925.npy",
         "FV patch → Built_up (JUDGE.md Built_up val case 7)"),
        ("data/patches_angola/Forest_Vegetation/patch_m14.86899_15.57579.npy",
         "FV patch → Built_up (JUDGE.md Built_up val case 10)"),
        ("data/patches_angola/Built_up/patch_m17.92213_19.76610.npy",
         "Bu patch → Wetland (JUDGE.md Built_up val case 1)"),
    ]

    GLOBAL6_CLASSES_LOCAL = [
        "Forest_Vegetation", "Cropland", "Water", "Built_up", "Bare_Sparse", "Wetland"
    ]

    for rel_path, desc in bad_patches:
        full_path = str(ROOT / rel_path)
        if not os.path.isfile(full_path):
            print(f"    [SKIP] {rel_path} — not found")
            continue
        arr    = np.load(full_path).astype(np.float32)
        arr    = np.clip(arr, 0.0, 1.0)
        tensor = _NORMALIZE(torch.from_numpy(arr)).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor).squeeze(0)
        probs_raw = F.softmax(logits,       dim=0).numpy()
        probs_cal = F.softmax(logits / T,   dim=0).numpy()
        pred_raw  = GLOBAL6_CLASSES_LOCAL[probs_raw.argmax()]
        pred_cal  = GLOBAL6_CLASSES_LOCAL[probs_cal.argmax()]
        conf_raw  = probs_raw.max()
        conf_cal  = probs_cal.max()
        print(f"    {desc}")
        print(f"      Before T: {pred_raw} {conf_raw*100:.1f}%")
        print(f"      After  T: {pred_cal} {conf_cal*100:.1f}%  (T={T:.4f})")


if __name__ == "__main__":
    main()
