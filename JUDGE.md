# Judge Notes — Wildfire Dashboard AI

This document records known limitations, confirmed bugs, and diagnostic
findings that a technical judge should be aware of when evaluating the
dashboard. Each entry is grounded in actual debug-log output or probe
results, not speculation.

---

## Land Cover Classifier (🌿 Land Cover tab)

### Model

MobileNetV2 fine-tuned on [EuroSAT](https://github.com/phelber/EuroSAT) —
27,000 satellite images across 10 classes (AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop,
Residential, River, SeaLake). Validated at **94% accuracy on real EuroSAT
test images** (30 images × 10 classes) when run through the current
inference pipeline.

Input: Sentinel-2 HLS bands B2/B3/B4 from NASA Earthdata, fetched
on-demand for any configured country.

---

### Fixed bugs — Phase A (no longer present)

#### 1. Unscaled HLS reflectances → 100% SeaLake on any input

**Root cause.** HLS reflectances are physical values in `[0, 0.25]`. The
original pipeline passed them through `arr * 255 → uint8 → PIL` without
rescaling, producing pixel values of DN 0–44. EuroSAT was trained on
imagery already stretched to the full `[0, 255]` uint8 range; DN 0–44 is
the model's deep-water (SeaLake) activation space. Every land-cover scene
predicted SeaLake at 100%.

**Fix.** Per-channel p2/p98 stretch of HLS float32 input before the PIL
conversion, applied only in the `else` (float32) branch of
`classify_tile()`. Nodata pixels (all-zero across all bands after
`nan_to_num`) are filled with the per-channel valid-pixel mean so the
model does not see an extreme bimodal black-hole/bright-land pattern.

**Verification.** Confirmed working: Angola granule G4254742468, date
2026-07-19 → **HerbaceousVegetation 63.9%**, SeaLake 0.0%.

---

#### 2. Water pixels selected as "best land crop" → wrong crop for Greece

**Root cause.** The crop-selection loop ranked windows by
`1 - np.isnan(B4).mean()`. Over-water pixels in HLS have `B4 = 0.0` (real
zero reflectance, not NaN), so they passed the NaN validity test. For
coastal/island countries (Greece, Aegean granules) the selector picked a
water-dominant window because it had 100% "valid" (non-NaN) pixels. After
the p2/p98 stretch, the image had R=34% nonzero/G=54%/B=100%, which no
EuroSAT class resembles.

**Fix.** Crop selection now counts pixels with `B4 > 0.001` (positive red
reflectance) rather than `~np.isnan(B4)`. This excludes open water and
deep shadows while accepting all real land reflectances (soil, vegetation,
urban surfaces all have B4 > 0.002 in HLS).

**Verification.** Greece now selects a land-dominant window (land_frac=94%)
instead of an Aegean water window.

---

### Known limitation — out-of-domain inference (not a bug)

#### Dense tropical canopy classified as Residential at high confidence

**Confirmed instance.** Angola granule G4177839477, acquisition date
2026-05-03.

Debug log (verbatim):
```
ch0(B4/R): min=0.008300  max=0.267200  mean=0.032876  nonzero=100.0%
ch1(B3/G): min=0.009400  max=0.255700  mean=0.045567  nonzero=100.0%
ch2(B2/B): min=0.005600  max=0.191500  mean=0.023060  nonzero=100.0%
crop_ndvi_mean=0.743  veg>0.3=99.8%
crop window: y=[98,1122] x=[98,1122]

[post-stretch p2/p98] — pre-model input range:
    ch0: mean DN=91.9   max DN=255  nonzero=97.7%
    ch1: mean DN=105.3  max DN=255  nonzero=98.0%
    ch2: mean DN=95.6   max DN=255  nonzero=97.9%
```

Model output: **Residential 95.0%**. NDVI=0.743, veg>0.3=99.8%.

**Why this happens.** EuroSAT Forest = European conifer forest, which has a
distinctive spectral signature: `B > G > R`, dark overall (mean DN ≈
40/64/76). Dense tropical Angola rainforest, after p2/p98 stretch, produces
`G > B > R` at mid-brightness (92/105/96 DN) — green-dominant, not
blue-dominant. The L2 distance from the post-stretch means to each EuroSAT
class centroid:

| Class | EuroSAT mean DN [R,G,B] | L2 from [92,105,96] |
|---|---|---|
| **Residential** | **[87, 94, 102]** | **13.7 ← closest** |
| Highway | [76, 91, 94] | 20.8 |
| HerbaceousVegetation | [119, 103, 105] | 29.0 |
| Forest | [40, 64, 76] | 69.0 |

The p2/p98 stretch works correctly — it maps HLS reflectances into the
EuroSAT DN range. But it cannot change where the mean of the stretched image
lands relative to EuroSAT class centroids. Tropical African rainforest has
no analog in EuroSAT.

**This is not a fixable code issue.** Fixing it would require retraining on
a dataset that includes tropical African land cover (e.g. BigEarthNet,
SEN12MS, or a purpose-built HLS dataset). No change to `classify_tile()` can
bridge a training-data gap.

**What to trust instead.** The NDVI metric (crop mean=0.743, 99.8% of
pixels > 0.3) is derived directly from the satellite bands and is
independent of the EuroSAT model. It is a reliable vegetation signal for
any region. The Land Cover tab displays it prominently above the model
result for exactly this reason.

**UI mitigation.** Non-European countries display this badge on the
classifier result:

> ⚠️ **Experimental** — classifier trained only on European land cover
> (EuroSAT); results outside Europe can be confidently wrong even at high
> confidence scores. Use the NDVI metric above as a more reliable
> vegetation signal for this region.

---

### Phase B normalisation bug — found and fixed during Phase B close-out

#### Built_up at 100% confidence on NDVI=0.603 vegetated tile

**Symptom.** Live Angola run: crop NDVI mean=0.603, veg>0.3=99.7% —
classification `Built_up` at 100% confidence. Same crop coords as the
original Fase A bug (y=[98,1122] x=[98,1122]).

**Root cause.** Two models in this codebase were trained with incompatible
normalisation pipelines, and `classify_tile()` had only one float32 branch:

| Model | Training normalisation |
|---|---|
| EuroSAT-10 (Fase A) | uint8 JPEG images, DN 0–255 → ImageNet Normalize |
| Global-6 (Phase B) | raw Sentinel-2 reflectance [0, ~0.25] → **ImageNet Normalize directly, no stretch** |

`classify_tile()` applied p2/p98 stretch to all float32 inputs, which was correct
for the EuroSAT-10 model (live HLS tiles need stretching to reach the DN range the
model was trained on). But for the Global-6 model this stretching moves a typical
reflectance value of `0.045` from normalized `−1.92` to `−1.73` — a shift of ~0.2
units, which is enough to move a vegetated scene from the Forest_Vegetation decision
region into Built_up.

**Evidence (direct torch inference):**

```
Same FV training patch, two paths:
  Training path (direct _NORMALIZE):       Forest_Vegetation  95.5%
  classify_tile() path (p2/p98 then norm): Built_up          100.0%

Live Angola tile (NDVI=0.603) via fixed path:
  Forest_Vegetation  60.9%   Built_up 27.6%   Wetland  9.9%
```

**Fix.** [`landcover_classifier.py`](landcover_classifier.py) now branches on `_model_version`:
- `global6`: fill nodata, clip to `[0,1]`, then `ToTensor → Resize → Normalize` (matches training)
- `eurosat10`: p2/p98 stretch as before (Fase A fix preserved)

**Verification.** After fix:
- FV val patches through `classify_tile()`: **6/10 correct** (≈ training-eval 7/10; 1-sample difference is patch orientation vs. training's CHW tensor path)
- Built_up val patches: **5/8 correct** (exact match to training eval)
- Live Angola tile: **Forest_Vegetation 60.9%** (was Built_up 100%)

---

### Geographic scope — Phase A (Fase A) classifier

| Coverage | Countries in this dashboard |
|---|---|
| ✅ Validated (EuroSAT in-domain) | Greece, Portugal |
| ⚠️ Experimental (out-of-domain) | All others — Angola, Brazil, Australia, DRC, Indonesia, Russia, etc. |

EuroSAT training data covers Europe and parts of the Middle East / North
Africa. The model was not trained on, and has not been validated against,
tropical, boreal, or Southern Hemisphere biomes.

---

## Phase B — Global-6 Classifier

### Global-6 schema and mapping tables

Six canonical classes replace the 10 EuroSAT-native classes for Phase B.
These are the single source of truth in [`landcover_schema.py`](landcover_schema.py).

**EuroSAT → Global-6 mapping**

| EuroSAT class | → Global-6 |
|---|---|
| Forest | Forest_Vegetation |
| HerbaceousVegetation | Forest_Vegetation |
| Pasture | Forest_Vegetation |
| AnnualCrop | Cropland |
| PermanentCrop | Cropland |
| SeaLake | Water |
| River | Water |
| Residential | Built_up |
| Industrial | Built_up |
| Highway | Built_up |

**ESA WorldCover v200 → Global-6 mapping**

| WorldCover code | Label | → Global-6 |
|---|---|---|
| 10 | Tree cover | Forest_Vegetation |
| 20 | Shrubland | Forest_Vegetation |
| 30 | Grassland | Forest_Vegetation |
| 100 | Moss & lichen | Forest_Vegetation |
| 40 | Cropland | Cropland |
| 80 | Permanent water bodies | Water |
| 50 | Built-up | Built_up |
| 60 | Bare / sparse vegetation | Bare_Sparse |
| 70 | Snow & ice | Bare_Sparse |
| 90 | Herbaceous wetland | Wetland |
| 95 | Mangroves | Wetland |

---

### Model (v2 — shipped checkpoint)

**File:** `models/global6_classifier.pt`
**mtime:** 2026-08-20 18:45:03 UTC−6 (matches Track 4 retrain completion)
**Architecture:** MobileNetV2, last 3 InvertedResidual blocks unfrozen
**Classes:** 6 (Global-6 schema above)
**Training data:**
- EuroSAT: 21,600 train / 5,400 val (80/20 stratified split, seed=42)
- Angola Sentinel-2 patches: 474 train / 118 val

---

### Phase B training — v1 vs v2 decision

**v2 is the shipped model. This is the authoritative decision record.**

**Why v2 was chosen over v1 despite lower Angola overall accuracy:**

v1 used class-level √-inverse-frequency loss weighting only. For
`Forest_Vegetation` and `Built_up` — ~99% EuroSAT by sample count —
Angola contributed only 0.5–0.6% of within-class gradient mass
(160–188× European dominance). Result: 9/10 Angola forest patches
were predicted as Wetland (the only class with an Angola-only gradient
boundary). This was the original Fase A bug transplanted into the new
schema: dense tropical vegetation misclassified at near-100% confidence
as a spectrally-adjacent but semantically wrong class.

v2 replaced pure loss-reweighting with a **source-aware two-factor
WeightedRandomSampler**:
- `sample_weight = base_class_w × source_boost`
- `source_boost` set so Angola FV/Built_up samples represent **25% of
  within-class gradient mass** (up from 0.5–0.6%)
- Angola FV images drawn ~40× per epoch; Built_up ~47×
- Stronger Angola augmentation (90° rotations, ±25% brightness/contrast
  jitter, 50%-probability box blur) to offset memorisation risk

The fix directly resolved the targeted failure: Forest_Vegetation
corrected 10% → 70%, Built_up 37.5% → 62.5%. Both exceed the 50%
threshold set as the go/no-go criterion before the retrain.

The cost — Cropland (80%→45%) and Water (70%→40%) — is a known accepted
trade-off, not an oversight. It arises because the WRS boost for FV/Built_up
displaces Cropland/Water epoch draws (~4,480 → ~4,009 each), and because
arid-region Angolan cropland shares mid-brightness spectral signatures with
the newly-enlarged Built_up boundary. No further training iterations are
planned: this is a deliberate stopping point at the hackathon deadline,
not a resource limitation being glossed over.

**No third retrain will be attempted.** The remaining Cropland/Water weakness
is documented below as a known limitation.

---

### v1 vs v2 accuracy comparison

**EuroSAT val (5,400 samples — regression guard for Greece/Portugal):**

| Class | v1 (loss-only) | v2 (WRS+aug) | Δ |
|---|---:|---:|---:|
| Forest_Vegetation | 0.9769 | 0.9688 | −0.008 |
| Cropland | 0.9518 | 0.9464 | −0.005 |
| Water | 0.9591 | 0.9527 | −0.006 |
| Built_up | 0.9587 | 0.9513 | −0.007 |
| **Overall** | **0.9628** | **0.9557** | **−0.007** |

No regression: all European classes drop < 1 pp. The drop is expected —
WRS shifts ~25% of epoch capacity to Angola FV/Built_up repetitions,
slightly reducing effective EuroSAT signal per epoch.

**Angola val (118 samples — generalisation check):**

| Class | n | v1 | v2 | Δ | Note |
|---|---:|---:|---:|---:|---|
| Forest_Vegetation | 10 | 0.10 | **0.70** | **+0.60** | Primary fix target ✅ |
| Cropland | 20 | 0.80 | 0.45 | −0.35 | Known regression ⚠️ |
| Water | 20 | 0.70 | 0.40 | −0.30 | Known regression ⚠️ |
| Built_up | 8 | 0.375 | **0.625** | **+0.25** | Secondary fix target ✅ |
| Bare_Sparse | 30 | 1.00 | 1.00 | 0.00 | ✅ |
| Wetland | 30 | 1.00 | 0.967 | −0.03 | ✅ |
| **Overall** | **118** | **0.797** | **0.746** | **−0.051** | Lower overall; better on targeted classes |

The lower overall Angola accuracy in v2 is a known, accepted trade-off.
v2 was chosen because it fixes the class that originally motivated Phase B.

---

### Angola v2 confusion matrix (rows = True, cols = Predicted)

```
True / Pred         Fv    Cr    Wa    Bu    Ba    We   n
──────────────────────────────────────────────────────────
Forest_Vegetation    7     0     0     2     0     1   10   ← 7/10 correct (was 1/10)
Cropland             3     9     0     8     0     0   20   ← regression; errors into Built_up
Water                0     0     8     7     2     3   20   ← regression; errors into Built_up
Built_up             2     0     0     5     0     1    8   ← 5/8 correct (was 3/8)
Bare_Sparse          0     0     0     0    30     0   30   ✓ perfect
Wetland              1     0     0     0     0    29   30   ✓ near-perfect
```

**Key finding — Cropland/Water errors concentrate into Built_up:**
Of 20 Cropland samples, 8 are predicted as Built_up (not random scatter).
Of 20 Water samples, 7 are predicted as Built_up. This is a spectral
overlap problem: in Angola's arid inland regions, seasonally-bare cropland
fields and low-density built-up settlements share similar mid-brightness
green/brown reflectance after p2/p98 normalisation. The WRS boost expanded
the Built_up decision boundary to accommodate Angolan settlement patterns,
and that boundary now over-extends into adjacent bare-soil farmland and
turbid/shallow water.

This is a **data-quantity limitation**, not a code bug. Fixing it would
require more diverse Angola Cropland and Water training patches (currently
80 each).

---

### Per-sample breakdown — Forest_Vegetation (n=10) and Built_up (n=8)

**Forest_Vegetation v2 (7/10 correct):**

| Patch | Result |
|---|---|
| `patch_m17.81863_22.09406.npy` | ✓ correct |
| `patch_m17.19766_19.30051.npy` | ✓ correct |
| `patch_m16.16269_22.61138.npy` | ✓ correct |
| `patch_m13.16130_21.88713.npy` | ✗ → Wetland |
| `patch_m10.62565_22.14579.npy` | ✓ correct |
| `patch_m9.12495_22.09406.npy`  | ✓ correct |
| `patch_m4.46762_15.67925.npy`  | ✗ → Built_up |
| `patch_m12.54033_23.02524.npy` | ✓ correct |
| `patch_m5.03685_14.28248.npy`  | ✓ correct |
| `patch_m14.86899_15.57579.npy` | ✗ → Built_up |

Wetland-spillover resolved: 6/9 wrong → Wetland (v1) reduced to 1/3 wrong → Wetland (v2).
Remaining 2 errors are FV↔Built_up at sparse woodland/settlement edges — genuine spectral ambiguity.

**Built_up v2 (5/8 correct):**

| Patch | Result |
|---|---|
| `patch_m17.92213_19.76610.npy` | ✗ → Wetland |
| `patch_m8.81446_13.29957.npy`  | ✓ correct |
| `patch_m8.91796_13.35130.npy`  | ✓ correct |
| `patch_m4.41587_15.42059.npy`  | ✓ correct |
| `patch_m8.81446_13.35130.npy`  | ✓ correct |
| `patch_m4.41587_15.36886.npy`  | ✓ correct |
| `patch_m5.91657_22.40445.npy`  | ✗ → Forest_Vegetation |
| `patch_m7.62426_15.05846.npy`  | ✗ → Forest_Vegetation |

Improved from 3/8 → 5/8. Remaining 3 errors: 1 Wetland, 2 Forest_Vegetation.
No longer collapsing into Wetland as in v1.

---

### Geographic scope — Phase B (Global-6) classifier

| Coverage | Countries in this dashboard | Per-class notes |
|---|---|---|
| ✅ Validated | Greece, Portugal | EuroSAT in-domain; all 4 active classes ≥ 94.6% |
| ✅ Improved experimental | Angola | Forest_Vegetation 70%, Built_up 63%, Bare_Sparse 100%, Wetland 97%; **Cropland 45% and Water 40% are weaker than the Fase A model** |
| ⚠️ Experimental (untested) | Brazil, DRC, Indonesia, Australia, Russia, etc. | No training or validation data; same domain-gap caveat as Fase A |

**Angola nuance:** v2 is meaningfully better than v1 for vegetation and
built-environment classification. It is **not** a general improvement —
Cropland and Water accuracy in Angola are lower than the original 10-class
model's proxy scores. For Angolan scenes, the NDVI metric displayed above
the classifier result remains the more reliable vegetation signal.

---

## ~~Known limitation — crop validity check does not filter cloud shadow~~ — RESOLVED ✅

> **Status: RESOLVED** — Fmask cloud filter implemented and verified.
> Fix committed: [`sentinel_fetch.py`](sentinel_fetch.py), [`app.py`](app.py),
> [`scripts/validate_countries.py`](scripts/validate_countries.py).

### Original symptom (preserved for reference)

Portugal live run, 17:39:51 UTC, granule with 25% cloud cover.
Crop NDVI mean=0.081, veg>0.3=8.9%, yet model predicted `Forest_Vegetation`
at 100% confidence. Root cause: crop validity check counted only `B4 > 0.001`,
which passes cloud tops (over-unity reflectance) and cloud shadow pixels alike.

### Fix implemented

**Step 1 — Read the Fmask band.**
[`sentinel_fetch.py`'s `_download_and_read()`](sentinel_fetch.py:408) now fetches
the HLS `Fmask` quality band alongside B2/B3/B4/B8:

- Strategy B (one file per band): matches files with `"FMASK"` in the filename.
- Strategy A (single multi-band file): `desc_to_key` extended with `"Fmask"/"FMASK"`.
- Fmask is returned as float32 **without** the `_HLS_SCALE` (×0.0001) applied —
  it is a QA integer band, not a reflectance band.
- The `is_cloud_contaminated()` helper auto-detects accidentally-scaled values
  (max < 1.0) and reverses the scaling before bit-testing.

**Step 2 — Fmask bit layout (source: HLS V2.0 User Guide, LP DAAC 2023,
Table 7, https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf):**

| Bit | Description |
|-----|-------------|
| 0 | Cirrus |
| **1** | **Cloud** |
| **2** | **Adjacent to cloud / cloud shadow** |
| **3** | **Cloud shadow** |
| 4 | Snow / ice |
| 5 | Water |
| 6–7 | Aerosol level |

Mask applied: `Fmask & 0b00001110 != 0` → bits 1, 2, 3 = cloud | adjacent | shadow.
(Note: the earlier JUDGE.md entry suggested `0b00110010` — that was incorrect; the
corrected mask from the actual HLS v2.0 User Guide is `0x0E = 0b00001110`.)

**Step 3 — Wired into crop selection.**
[`app.py`](app.py:1488) now builds `_cloud_mask = sf.is_cloud_contaminated(fmask_arr)`
and uses it in the 5×5 grid scan. A pixel is valid only if:
- `B4 > 0.001` (positive reflectance — excludes nodata/water), **AND**
- `NOT cloud-contaminated` (Fmask bits 1/2/3 clear).

The fallback path (argmax of row/col sums) is also updated. If `row_valid.max() == 0`
after cloud filtering — meaning no cloud-free land pixels exist anywhere in the scene —
the app issues an honest warning and stops rather than falling back silently to a
cloud-contaminated crop.

[`scripts/validate_countries.py`](scripts/validate_countries.py) mirrors the same
cloud-filter logic in its `best_crop()` function.

### Verification — known bad case (Portugal)

**Granule fetched:** `G4174286832-LPCLOUD`, 2026-04-30, **45% cloud cover**.

```
Fmask shape: (3660, 3660)  dtype=float32
Fmask unique values: [64, 66, 68, 70, 72, 74, 76, 78, 80, 96, 100, 112, 128,
                      130, 132, 134, 136, 138, 140, 142, 144, 148, 160, 164,
                      176, 192, 194, 196, 198, 200, 202, 204, 206, 208, 212,
                      224, 228, 240, 255]
Pixels flagged cloud/shadow (Fmask): 64.2%
```

Bit decode confirms correct operation:
- `64 = 01000000` = low-aerosol, clear pixel → **not flagged** ✅
- `66 = 01000010` = cloud (bit 1) → **flagged** ✅
- `68 = 01000100` = adjacent-to-cloud (bit 2) → **flagged** ✅
- `78 = 01001110` = cloud+adjacent+shadow (bits 1,2,3) → **flagged** ✅
- `255` = nodata → **flagged** ✅

**Before/after on the previously bad crop window** `y=[98,1122] x=[2538,3562]`:

| | Old (B4-only) | New (B4 + Fmask) |
|---|---|---|
| Valid pixel % | **98.3%** | **30.5%** |
| Drop | — | **−67.8 pp** |

The old 98.3% "valid" fraction was entirely driven by cloud pixels passing the
`B4 > 0.001` threshold. The Fmask filter correctly reduces this to 30.5%.

**New best crop (Fmask-filtered):** `y=[1928,2952] x=[2538,3562]`, valid=**63.8%**
(cloud-free land pixels only).

### Regression tests — clear-sky scenes

| Country | Granule | Cloud% | Old valid% | New valid% | Drop | Result |
|---|---|---|---|---|---|---|
| Portugal | G4171495522 | 3.0% | 100.0% | 99.6% | 0.4 pp | ✅ Clear scene: minimal removal |
| Greece | G4120179113 | 2.0% | 100.0% | 100.0% | 1.5 pp | ✅ Clear scene: minimal removal |
| Angola | G4117886341 | 19.0% | 40.0% | 29.1% | 10.9 pp | ✅ Best crop: 90.8% cloud-free valid |

Angola has 19% cloud (the CMR metadata cloud figure), but the best selected crop
window with the Fmask filter still achieves **90.8% valid land pixels** — the crop
selector correctly navigated to the cloud-free portion of the tile.

### Task 4 re-examination of flagged tiles

Four tiles previously flagged as implausible were re-fetched with Fmask active:

| Country/Tile | Previously flagged as | Fmask cloud-flagged px | Old valid% | New valid% | Drop | Diagnosis |
|---|---|---|---|---|---|---|
| India Tile 4 | Cloud contamination (NDVI=−0.072) | 47.9% | 51.5% | 50.5% | 1.0 pp | **Domain gap**, not cloud — best crop valid=98.0% with NDVI=−0.073 unchanged |
| Australia Tile 1 | Cloud/sensor artifact (NDVI=−0.988) | 0.7% | 98.0% | 98.0% | 0.0 pp | **Sensor artifact / bare rock** — Fmask shows clear sky; NDVI still −0.981 |
| Brazil Tile 3 | Domain gap (Built_up on NDVI=0.838) | 0.0% | 100.0% | 100.0% | 0.0 pp | Confirmed **domain gap** — no cloud involvement |
| Mexico Tile 2 | Semi-arid failure (NDVI=0.039) | 7.1% | 90.8% | 88.3% | 2.5 pp | Confirmed **domain gap** — semi-arid bare soil, not cloud |

**Key finding:** The India Tile 4 case, previously labeled "almost certainly cloud
contamination," is **not** explained by Fmask. The Fmask band shows 47.9% of scene
pixels are cloud-flagged, but those are in a different part of the tile — the best
selected crop has 98.0% valid land pixels with NDVI=−0.073 **unchanged** after
filtering. The sub-zero NDVI on clear-sky land is a genuine spectral anomaly,
likely bare rock / salt flat / radiometric calibration artefact in that sub-region
of India. The JUDGE.md note "almost certainly cloud" was over-confident — the Fmask
evidence does not support it.

**Updated Task 4 buckets (post-Fmask):** No change — the original bucket assignments
(Brazil=Plausible, India=Mixed, Australia=Mixed, Mexico=Unreliable) remain correct.
The failures are confirmed domain gaps or spectral anomalies, not cloud contamination
that the Fmask filter could address.

### Residual limitations

1. **Fmask unavailable (Strategy A — single-file granule):** When `earthaccess.open()`
   returns a single multi-band file, the Fmask band is only loaded if its subdataset
   description is `"Fmask"` or `"FMASK"`. If the HLS file uses a different subdataset
   name, the band is silently skipped and the system falls back to B4-only filtering.
   In practice, HLS S30 v2.0 granules are delivered as one file per band (Strategy B),
   so this path is rarely exercised. If Fmask is `None`, the B4-only criterion still
   applies — no regression.

2. **Fully cloud-covered granule with no clear pixels:** If `row_valid.max() == 0`
   after Fmask filtering, the app now issues a warning: *"No cloud-free land pixels
   found in this granule. Try a different date range."* This is the honest-refusal
   path — not implemented before this fix. It was not triggered in any of the test
   runs (the worst case, 45% cloud cover, still had 63.8% valid-pixel crops available
   in a different part of the tile).

3. **NDVI-based diagnosis remains valid complement:** For scenes where Fmask is
   unavailable or granule-level cloud cover is low but spatially concentrated, the
   NDVI (mean < 0.1 with veg>0.3 < 5%) remains a reliable secondary indicator that
   the selected crop is cloud-contaminated.

---

## Task 2 — Confidence calibration (temperature scaling)

### Methodology

Standard temperature scaling (Guo et al. 2017 "On Calibration of Modern Neural
Networks"). A single scalar T is fitted per model by minimising negative
log-likelihood on the held-out validation logits (not softmax probabilities —
raw logits) using LBFGS. At inference, `softmax(logits / T)` replaces
`softmax(logits)`. T > 1 softens overconfident predictions; T < 1 would
sharpen under-confident ones (did not occur here).

**Script:** [`scripts/calibrate_temperature.py`](scripts/calibrate_temperature.py)
**Output files:** `models/global6_temperature.json`, `models/eurosat10_temperature.json`

### Fitted T values

| Model | T | Fit data | Note |
|---|---:|---|---|
| `global6_classifier.pt` | **1.0333** | EuroSAT val (5,400) + Angola val (118) — combined | Close to 1.0 — model is already well-calibrated at aggregate ECE level |
| `landcover_classifier.pt` (EuroSAT-10) | **1.1101** | EuroSAT val (5,400) only | Slightly more overconfident |

Both T values are single scalars, not per-class. Per-class scaling was
considered but not used because: (a) a single T is the standard recipe and is
sufficient given the small T values observed, and (b) the Angola val set has
only 118 samples — fitting 6 per-class temperatures would overfit on 10–30
samples per class.

### ECE before/after

Expected Calibration Error (15-bin equal-width, Guo et al.):

| Model | Dataset | ECE before | ECE after | Δ |
|---|---|---:|---:|---:|
| global6  | EuroSAT val (5,400) | 0.0052 | 0.0065 | +0.0013 |
| global6  | Angola val (118)    | 0.1689 | 0.1669 | −0.0020 |
| eurosat10 | EuroSAT val (5,400) | 0.0103 | 0.0120 | +0.0017 |

**Key finding — calibration had minimal effect on the specific
high-confidence-wrong cases this session has repeatedly flagged:**

Both models are already very well calibrated at the aggregate ECE level
(global6 ECE = 0.0052 on EuroSAT val — excellent). T=1.0333 for global6 is
near-unity, meaning the softmax output barely changes. Confidence reductions
on the known bad cases are 0.1–1.0 pp — statistically real but
operationally negligible. **The 100%-confidence-on-wrong-answer problem is
not solved by this calibration.** It cannot be: the cause is a training-data
domain gap (no African or semi-arid biomes in EuroSAT), and temperature
scaling cannot compensate for a missing class. It only rescales the logit
magnitude — it has nothing to rescale toward if the correct class is absent
from the vocabulary entirely.

This is worth stating explicitly rather than burying: **a judge who sees
"temperature scaling implemented" should not conclude that the overconfidence
problem is fixed. It is not. The NDVI metric displayed above the classifier
result in the UI remains the primary trust signal for any non-European,
non-Angolan scene.**

### Known bad cases — confidence before/after T

These are the specific patches from JUDGE.md Phase B, re-run through the
production `classify_tile()` pipeline (global6 path, T=1.0333):

| Patch | True class | Predicted | Before T | After T | Δ |
|---|---|---|---:|---:|---:|
| `patch_m4.46762_15.67925` | Forest_Vegetation | Built_up | 63.6% | 63.0% | −0.6 pp |
| `patch_m14.86899_15.57579` | Forest_Vegetation | Built_up | 78.3% | 77.3% | −1.0 pp |
| `patch_m17.92213_19.76610` | Built_up | Wetland | 99.9% | 99.8% | −0.1 pp |

The prediction does not change. Wrong predictions remain wrong.
The near-zero Δ for the Wetland case (0.1 pp) is the starkest illustration:
T=1.0333 divides an already-extreme logit by 1.03, which is imperceptible at
the 99.9% confidence level. That specific failure requires either more
training data (Angolan built-up settlement patches in the training set) or a
different model architecture — not temperature scaling.

The earlier "91.1%" figure from the calibration fitting script reflects a
different preprocessing path (direct tensor, no PIL roundtrip) vs. the
production `classify_tile()` path (HWC float32 → PIL uint8 → ToTensor).
The production path gives 63.6% raw / 63.0% calibrated on that specific patch.

### Wiring

T is loaded in `load_landcover_model()` from `models/global6_temperature.json`
or `models/eurosat10_temperature.json` and stored in the module-level
`_TEMPERATURE` float. `classify_tile()` applies `F.softmax(logits / _TEMPERATURE, dim=1)`
instead of plain `F.softmax(logits, dim=1)`. Each model gets its own
independently-fitted T. The returned dict now includes `"temperature": T`
for traceability.

---

## Task 4 — Lightweight multi-country validation

### Methodology

**Explicit caveat: this is not equivalent to Phase B's real validation.**
Phase B fine-tuned the model on 474 labelled Angola patches and evaluated on
118 held-out patches with known ground-truth labels. Task 4 uses zero
ground-truth labels — it is a qualitative NDVI-consistency check only. A
prediction is "plausible" if the predicted land-cover class is physically
consistent with the tile's NDVI value; "implausible" if they contradict
each other; "borderline" if the NDVI is in an ambiguous range.

**Script:** [`scripts/validate_countries.py`](scripts/validate_countries.py)
**Report:** [`reports/country_validation_task4.json`](reports/country_validation_task4.json)

### Country selection

Four countries chosen for biome diversity:

| Country | Biome rationale | Expected primary class |
|---|---|---|
| Brazil | Amazon tropical rainforest | Forest_Vegetation (NDVI > 0.5) |
| India | Indo-Gangetic mixed agriculture + monsoon season | Cropland (moderate NDVI) |
| Australia | Central/NW arid scrub + SE temperate | Bare_Sparse (NDVI < 0.2) |
| Mexico | Mixed subtropical/semi-arid | Diverse |

### Fetch times

5 tiles per country, real Sentinel-2/HLS (HLSS30 v2.0) via NASA Earthdata:

| Country | Tiles fetched | Total fetch time | Per-tile avg |
|---|---:|---:|---:|
| Brazil | 5/5 | 167 s | 33 s |
| India | 5/5 | 178 s | 36 s |
| Australia | 5/5 | 183 s | 37 s |
| Mexico | 4/5 (1 timeout) | 177 s | 44 s |

All well within the 0.5–1 day budget (total: ~12 minutes for 19 successful fetches).

### Per-tile results

**Brazil** (bbox: −73.99,−33.75,−28.85,5.27)

| Tile | Granule date | Cloud % | NDVI mean | Veg>0.3 | Predicted | Conf | NDVI consistent? |
|---:|---|---:|---:|---:|---|---:|---|
| 1 | 2026-05-28 | 0% | 0.358 | 34% | Forest_Vegetation | 52.7% | ✅ plausible |
| 2 | 2026-06-19 | 0% | 0.754 | 99.9% | Forest_Vegetation | 79.5% | ✅ plausible |
| 3 | 2026-05-28 | 0% | 0.838 | 99.3% | **Built_up** | **81.3%** | ❌ implausible |
| 4 | 2026-05-28 | 0% | 0.713 | 56% | Forest_Vegetation | 65.8% | ✅ plausible |
| 5 | 2026-05-28 | 0% | 0.854 | 99.4% | Forest_Vegetation | 92.8% | ✅ plausible |

Tile 3 shows the same failure mode documented in JUDGE.md for Angola: dense
tropical vegetation (NDVI=0.838) predicted as Built_up. This is the Amazon
domain-gap — the model has no training signal for dense Neotropical canopy.

**India** (bbox: 68.18,8.07,97.40,35.51)

| Tile | Granule date | Cloud % | NDVI mean | Veg>0.3 | Predicted | Conf | NDVI consistent? |
|---:|---|---:|---:|---:|---|---:|---|
| 1 | 2026-05-29 | 0% | 0.475 | 15% | Forest_Vegetation | 54.3% | ✅ plausible |
| 2 | 2026-06-23 | 2% | 0.514 | 63% | Water | 70.6% | 🟡 borderline |
| 3 | 2026-05-28 | 0% | 0.204 | 8% | Wetland | 63.5% | 🟡 borderline |
| 4 | 2026-05-30 | 1% | −0.072 | 0.6% | **Forest_Vegetation** | **99.9%** | ❌ implausible |
| 5 | 2026-05-28 | 0% | 0.320 | 40% | Forest_Vegetation | 53.6% | 🟡 borderline |

Tile 4: NDVI=−0.072 is non-physical for land (below-water reflectance).
Re-examined with Fmask active (2026-07 re-run): Fmask flags 47.9% of scene
pixels as cloud/shadow, **but** those are in a different spatial portion of
the tile — the crop selector navigates to a 98.0% valid-land crop with NDVI
= −0.073 **unchanged** after cloud filtering. The sub-zero NDVI is a genuine
spectral anomaly (bare rock / salt flat / radiometric artefact) on clear-sky
land, not a cloud-contamination artifact. The original "almost certainly cloud"
diagnosis was incorrect; corrected to spectral anomaly / domain-gap failure.
Forest_Vegetation predicted at 99.9% remains a calibration failure (extreme
logit, T=1.03 insufficient), but the input crop is clean.

**Australia** (bbox: 113.34,−43.64,153.57,−10.68)

| Tile | Granule date | Cloud % | NDVI mean | Veg>0.3 | Predicted | Conf | NDVI consistent? |
|---:|---|---:|---:|---:|---|---:|---|
| 1 | 2026-06-07 | 0% | −0.988 | 0% | **Forest_Vegetation** | **48.4%** | ❌ implausible |
| 2 | 2026-05-29 | 2% | 0.361 | 80% | Forest_Vegetation | 92.2% | ✅ plausible |
| 3 | 2026-05-28 | 0% | 0.185 | 0.2% | Forest_Vegetation | 59.7% | 🟡 borderline |
| 4 | 2026-05-28 | 0% | 0.173 | 2% | Cropland | 60.7% | ✅ plausible |
| 5 | 2026-05-28 | 0% | 0.272 | 26% | Cropland | 51.5% | ✅ plausible |

Tile 1: NDVI=−0.988 is non-physical. Re-examined with Fmask active (2026-07
re-run): Fmask flags only 0.7% of pixels as cloud/shadow — the scene is clear.
Confirmed bare rock / desert sensor artefact, not cloud contamination.
The NDVI=−0.981 is unchanged with Fmask active (same granule, same crop).
Tiles 4–5 show appropriate Cropland predictions for SE Australia's wheat belt
(moderate NDVI ≈ 0.17–0.27).

**Mexico** (bbox: −117.13,14.53,−86.70,32.72)

| Tile | Granule date | Cloud % | NDVI mean | Veg>0.3 | Predicted | Conf | NDVI consistent? |
|---:|---|---:|---:|---:|---|---:|---|
| 1 | — | — | — | — | — | — | FETCH ERROR (connection reset) |
| 2 | 2026-05-29 | 2% | 0.039 | 2% | **Forest_Vegetation** | **68.1%** | ❌ implausible |
| 3 | 2026-05-30 | 0% | 0.129 | 0.2% | **Forest_Vegetation** | **92.8%** | ❌ implausible |
| 4 | 2026-07-20 | 1% | 0.554 | 57% | Forest_Vegetation | 54.0% | ✅ plausible |
| 5 | 2026-05-30 | 0% | 0.135 | 0.7% | **Forest_Vegetation** | **85.1%** | ❌ implausible |

Tiles 2, 3, 5 show Forest_Vegetation predictions on semi-arid Mexican scenes
with NDVI < 0.14.

#### The semi-arid failure mode is distinct from — and arguably worse than — the dense-vegetation failure

The Angola/Brazil failure pattern (dense tropical canopy predicted as Built_up)
and the Mexico failure pattern (bare semi-arid soil predicted as Forest_Vegetation)
look like variations on the same theme — "wrong class, out-of-domain" — but they
work differently and have different practical implications.

In Angola and Brazil the model sees a dense, high-NDVI scene and picks the
EuroSAT class whose spectral centroid is closest in the normalised feature
space. It confidently picks the wrong class (Built_up instead of
Forest_Vegetation), but NDVI > 0.5 immediately contradicts the prediction
and the user has a clear diagnostic. The Angola fine-tune directly addresses
this by teaching the model what Angolan vegetation looks like.

In Mexico (and by extension: United States SW, Bolivia, Argentina, South
Africa, Nigeria dry savanna, Russia steppe) the failure mechanism is
different. Dry scrub and bare soil have near-zero NDVI (0.04–0.14) and flat
reflectance across all three bands after the global6 normalisation path. The
model's vocabulary contains exactly one class that spans low-reflectance, low-
texture scenes with no clear spectral peak: in practice Forest_Vegetation acts
as the "catch-all" for scenes the model has never seen, because its decision
boundary is the widest and most diffuse after the WRS training that
de-emphasised non-Angola EuroSAT classes. The result is high-confidence
Forest_Vegetation (68–93%) on scenes that are emphatically not forest — and
crucially, the NDVI < 0.14 correctly contradicts this, but the model has no
way to flag its own uncertainty because it is not uncertain — it is confidently
predicting within a vocabulary that does not contain the right answer.

This failure is qualitatively harder to mitigate than the Angola case. The
Angola fine-tune added the correct spectral signature to the training set; no
equivalent fix exists for arid biomes without adding a representative set of
Bare_Sparse training patches from those regions. Until then, any country with
dry-season bare-soil land cover — a large subset of this dashboard's untested
countries — should be treated as at least as unreliable as Mexico, not merely
"Experimental" in the abstract sense.

### Country buckets

| Country | Bucket | P/B/I | Assessment |
|---|---|---|---|
| Brazil | **Plausible** | 4/0/1 | Mostly reliable for Amazonian forest; 1 known Built_up domain-gap failure |
| India | **Mixed** | 1/3/1 | High borderline rate — seasonal and cloud variability confounds model |
| Australia | **Mixed** | 3/1/1 | SE temperate zone plausible; inland/cloud-edge tiles unreliable |
| Mexico | **Unreliable** | 1/0/3 | Semi-arid Mexico strongly out-of-domain; Forest_Veg predicted on bare soil |

### Bucket definitions

- **Plausible**: > 70% of tiles are NDVI-consistent with prediction
- **Mixed**: < 70% plausible and < 50% implausible
- **Unreliable**: > 50% of tiles show NDVI contradiction

### Coverage update

| Coverage | Countries in this dashboard |
|---|---|
| ✅ Validated | Greece, Portugal |
| 🔵 Improved experimental | Angola |
| 🟡 Lightly checked — Plausible | Brazil |
| 🟡 Lightly checked — Mixed | India, Australia |
| 🟡 Lightly checked — Unreliable | Mexico |
| ⚠️ Experimental (untested) | United States, Canada, Indonesia, Russia, DRC, Mozambique, Bolivia, Venezuela, Argentina, China, Nigeria, South Africa, Chile |

**This lightweight check is explicitly lower-confidence than Phase B's real
fine-tuning + held-out accuracy measurement.** The 🟡 badge in the UI
acknowledges the review without overstating it.

