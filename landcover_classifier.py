"""
Inference wrapper for the land cover classifier.

Phase B: loads the Global-6 MobileNetV2 from models/global6_classifier.pt
(6 classes: Forest_Vegetation, Cropland, Water, Built_up, Bare_Sparse, Wetland).

Falls back to the Fase A 10-class EuroSAT model (models/landcover_classifier.pt)
if the Global-6 file is absent, so existing usage is never broken.

Usage:
    model = load_landcover_model()
    result = classify_tile(model, tile_array)
    # result: {"class": str, "confidence": float, "all_probs": dict,
    #          "model_version": "global6" | "eurosat10"}

NORMALISATION NOTE — float32 input path
----------------------------------------
There are two models in play, each trained with a different normalisation:

  Global-6 (Phase B):
      Training used raw Sentinel-2 surface reflectance values in [0, ~0.25]
      fed DIRECTLY to ImageNet Normalize(mean=[0.485,...], std=[0.229,...]).
      No per-tile stretch was applied during training.
      Inference MUST follow the same path: clip to [0,1], then Normalize.
      Applying p2/p98 stretch before Normalize shifts the input distribution
      by ~0.2 normalised units — enough to flip Built_up to Forest_Vegetation.

  EuroSAT-10 (Fase A):
      Training used EuroSAT JPEG images (uint8, full DN range).
      Live HLS tiles are raw reflectance [0,~0.25], which after ×255 land
      at DN 0–44 — deep inside the model's SeaLake activation space.
      Fix: p2/p98 stretch maps the physical reflectance range into [0,1]
      before the uint8 conversion, restoring the correct DN distribution.

Rule: float32 input + Global-6 model → direct clip + Normalize (no stretch).
      float32 input + EuroSAT-10 model → p2/p98 stretch (Fase A fix).
      uint8 input (any model) → /255 only, no stretch needed.
"""

import os
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Class lists
# ---------------------------------------------------------------------------

CLASSES_EUROSAT10 = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

CLASSES_GLOBAL6 = [
    "Forest_Vegetation",
    "Cropland",
    "Water",
    "Built_up",
    "Bare_Sparse",
    "Wetland",
]

# Active class list — set at load time, updated when the model loads
CLASSES = CLASSES_GLOBAL6   # default expectation; overridden in load_landcover_model()

_GLOBAL6_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "global6_classifier.pt"
)
_EUROSAT10_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "landcover_classifier.pt"
)

# Temperature-scaling scalar T fitted per model (temperature scaling, Guo et al. 2017).
# Loaded from models/*_temperature.json at load_landcover_model() time.
# softmax(logits / T) is applied instead of plain softmax(logits).
# T > 1 softens probabilities; falls back to T=1.0 (no scaling) if file is absent.
_TEMPERATURE: float = 1.0

# Module-level flag set when the Global-6 model is loaded
_model_version: str = "unknown"


def load_landcover_model(path: Optional[str] = None):
    """
    Load the land cover model.

    Preference order (when path is None):
      1. models/global6_classifier.pt  → Global-6, 6 classes  (Phase B)
      2. models/landcover_classifier.pt → EuroSAT-10, 10 classes (Fase A fallback)

    Returns the model in eval mode on CPU, or None if no file is found.
    Sets the module-level CLASSES list to match the loaded model.
    """
    global CLASSES, _model_version, _TEMPERATURE
    import json as _json
    import torch
    import torch.nn as nn
    from torchvision import models

    candidates = (
        [(path, None)] if path is not None
        else [
            (_GLOBAL6_MODEL_PATH,  "global6"),
            (_EUROSAT10_MODEL_PATH, "eurosat10"),
        ]
    )

    for fpath, version_hint in candidates:
        if not os.path.isfile(fpath):
            continue

        # Determine num_classes from file path / hint
        if version_hint == "global6" or (
            version_hint is None and "global6" in os.path.basename(fpath)
        ):
            num_classes = 6
            classes     = CLASSES_GLOBAL6
            ver         = "global6"
            temp_file   = os.path.join(os.path.dirname(__file__), "models", "global6_temperature.json")
        else:
            num_classes = 10
            classes     = CLASSES_EUROSAT10
            ver         = "eurosat10"
            temp_file   = os.path.join(os.path.dirname(__file__), "models", "eurosat10_temperature.json")

        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        state = torch.load(fpath, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        CLASSES = classes
        _model_version = ver

        # Load temperature scalar from JSON (falls back to 1.0 if absent)
        _TEMPERATURE = 1.0
        if os.path.isfile(temp_file):
            try:
                with open(temp_file) as _tf:
                    _TEMPERATURE = float(_json.load(_tf).get("T", 1.0))
                print(
                    f"[landcover_classifier] Temperature T={_TEMPERATURE:.4f} "
                    f"loaded from {temp_file}"
                )
            except Exception as _te:
                print(f"[landcover_classifier] Could not load temperature ({_te}); T=1.0")
        else:
            print(f"[landcover_classifier] No temperature file found ({temp_file}); T=1.0")

        print(f"[landcover_classifier] Loaded {ver} model ({num_classes} classes) from {fpath}")
        return model

    print(f"[landcover_classifier] No model file found (tried {[c[0] for c in candidates]})")
    return None


_DEBUG_LOG = "/tmp/classify_tile_debug.log"


def classify_tile(model, tile_array: np.ndarray) -> dict:
    """
    Classify a single satellite tile.

    Parameters
    ----------
    model : torch.nn.Module returned by load_landcover_model()
        Pass None to get a graceful error dict instead of a crash.
    tile_array : np.ndarray
        Shape (H, W, 3) or (3, H, W), dtype uint8 or float32.
        Values in [0, 255] (uint8) or [0.0, 1.0] (float32).

    Returns
    -------
    dict with keys:
        "class"         - predicted class name (str)
        "confidence"    - probability of the top class (float, 0–1)
        "all_probs"     - {class_name: probability} for all classes (dict)
        "model_version" - "global6" or "eurosat10"
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    # ── Debug logging ────────────────────────────────────────────────────────
    import time as _time
    _ts = _time.strftime("%H:%M:%S")
    try:
        arr_in = np.array(tile_array)
        with open(_DEBUG_LOG, "a") as _f:
            _f.write(f"\n[{_ts}] classify_tile called  model_version={_model_version}\n")
            _f.write(f"  input shape: {arr_in.shape}  dtype: {arr_in.dtype}\n")
            if arr_in.ndim == 3:
                for _ci in range(arr_in.shape[-1] if arr_in.shape[0] != 3 else arr_in.shape[0]):
                    if arr_in.shape[0] == 3:
                        _ch = arr_in[_ci]
                    else:
                        _ch = arr_in[:, :, _ci]
                    _f.write(
                        f"  ch{_ci}: min={float(_ch.min()):.6f}  max={float(_ch.max()):.6f}"
                        f"  mean={float(_ch.mean()):.6f}  nonzero={100*float((_ch != 0).mean()):.1f}%\n"
                    )
    except Exception as _dbg_err:
        try:
            open(_DEBUG_LOG, "a").write(f"[{_ts}] debug logging error: {_dbg_err}\n")
        except Exception:
            pass

    active_classes = CLASSES  # snapshot at call time

    if model is None:
        return {
            "class": None,
            "confidence": 0.0,
            "all_probs": {c: 0.0 for c in active_classes},
            "model_version": _model_version,
            "error": "Model not loaded — file missing or load failed.",
        }

    # ── Normalise array shape to (H, W, 3) ──────────────────────────────────
    arr = np.array(tile_array)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))   # (3,H,W) → (H,W,3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected shape (H,W,3) or (3,H,W), got {arr.shape}")

    # ── Convert to float32 [0,1] ─────────────────────────────────────────────
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if _model_version == "global6":
            # Global-6 model was trained on raw Sentinel-2 reflectance values
            # in [0, ~0.25] fed directly to ImageNet Normalize — no per-tile
            # stretch was applied during training.  Live HLS tiles have the
            # same physical scale, so we must follow the same path: clip only.
            # Applying p2/p98 stretch here would shift the normalised inputs by
            # ~0.2 units relative to what the model learned, collapsing
            # Forest_Vegetation → Built_up on every vegetated Angolan tile.
            nodata_mask = (arr.sum(axis=2) == 0.0)
            # Fill nodata pixels with per-channel mean of valid pixels so the
            # model sees smooth texture rather than extreme black holes.
            for _c in range(arr.shape[2]):
                valid_px = arr[:, :, _c][~nodata_mask]
                fill_val = float(valid_px.mean()) if valid_px.size > 0 else 0.0
                arr[nodata_mask, _c] = fill_val
            arr = np.clip(arr, 0.0, 1.0)
        else:
            # EuroSAT-10 (Fase A) model: trained on uint8 JPEG imagery with
            # full DN range.  HLS raw reflectances (~0.01–0.25) land at DN 0–44
            # without rescaling, putting every scene in the SeaLake activation
            # space.  Per-channel p2/p98 stretch restores the correct distribution.
            nodata_mask = (arr.sum(axis=2) == 0.0)
            arr_stretched = np.empty_like(arr)
            channel_means = []
            for _c in range(arr.shape[2]):
                ch = arr[:, :, _c]
                valid_px = ch[~nodata_mask]
                if valid_px.size > 0:
                    p2  = float(np.percentile(valid_px, 2))
                    p98 = float(np.percentile(valid_px, 98))
                    denom = p98 - p2
                    if denom < 1e-9:
                        stretched = np.where(~nodata_mask, ch / max(ch.max(), 1e-9), 0.0)
                    else:
                        stretched = np.where(
                            ~nodata_mask,
                            np.clip((ch - p2) / denom, 0.0, 1.0),
                            0.0,
                        )
                    valid_mean = float(stretched[~nodata_mask].mean())
                else:
                    stretched = np.zeros_like(ch)
                    valid_mean = 0.0
                arr_stretched[:, :, _c] = stretched
                channel_means.append(valid_mean)
            for _c, mean_val in enumerate(channel_means):
                arr_stretched[nodata_mask, _c] = mean_val
            arr = arr_stretched

    # ── Log pre-model stats ───────────────────────────────────────────────────
    try:
        with open(_DEBUG_LOG, "a") as _f:
            path_label = "direct clip (global6)" if _model_version == "global6" else "p2/p98 stretch (eurosat10)"
            _f.write(f"  [pre-model input, {path_label}]:\n")
            for _ci in range(arr.shape[2]):
                _sch = arr[:, :, _ci]
                _f.write(
                    f"    ch{_ci}: mean={float(_sch.mean()):.4f}  "
                    f"max={float(_sch.max()):.4f}  "
                    f"nonzero={100*float((_sch != 0).mean()):.1f}%\n"
                )
    except Exception:
        pass

    # ── Forward pass ─────────────────────────────────────────────────────────
    preprocess = transforms.Compose([
        transforms.ToTensor(),          # (H,W,3) → (3,H,W), already [0,1]
        transforms.Resize((64, 64)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    from PIL import Image
    pil_img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    tensor  = preprocess(pil_img).unsqueeze(0)   # (1, 3, 64, 64)

    with torch.no_grad():
        logits = model(tensor)
        # Temperature scaling: softmax(logits / T) instead of softmax(logits).
        # T is fitted per model on held-out val logits (NLL minimisation, LBFGS).
        # T > 1 softens overconfident probabilities; T=1.0 means no scaling.
        probs  = F.softmax(logits / _TEMPERATURE, dim=1).squeeze(0).numpy()

    top_idx = int(probs.argmax())
    return {
        "class":         active_classes[top_idx],
        "confidence":    float(probs[top_idx]),
        "all_probs":     {c: float(p) for c, p in zip(active_classes, probs)},
        "model_version": _model_version,
        "temperature":   _TEMPERATURE,
    }
