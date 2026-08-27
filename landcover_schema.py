"""
landcover_schema.py — Global 6-class land-cover scheme shared by EuroSAT and
ESA WorldCover sources.

The six canonical classes are:
    Forest_Vegetation — all tree cover, shrubland, grassland, herbaceous veg
    Cropland          — annual and permanent crops
    Water             — lakes, rivers, permanent open water
    Built_up          — residential, industrial, roads/highways
    Bare_Sparse       — bare land, snow & ice, moss/lichen
    Wetland           — herbaceous wetlands, mangroves

These mappings are the single source of truth for both:
  - Track 2 (EuroSAT relabelling, no network needed)
  - Track 3 (WorldCover-labelled Sentinel-2 patches, Angola + Brazil)
"""

# ---------------------------------------------------------------------------
# EuroSAT → Global-6
# ---------------------------------------------------------------------------

EUROSAT_TO_GLOBAL6: dict[str, str] = {
    "Forest":                 "Forest_Vegetation",
    "HerbaceousVegetation":   "Forest_Vegetation",
    "Pasture":                "Forest_Vegetation",
    "AnnualCrop":             "Cropland",
    "PermanentCrop":          "Cropland",
    "SeaLake":                "Water",
    "River":                  "Water",
    "Residential":            "Built_up",
    "Industrial":             "Built_up",
    "Highway":                "Built_up",
}

# ---------------------------------------------------------------------------
# ESA WorldCover v200 class codes → Global-6
# ---------------------------------------------------------------------------

WORLDCOVER_TO_GLOBAL6: dict[int, str] = {
    10:  "Forest_Vegetation",   # Tree cover
    20:  "Forest_Vegetation",   # Shrubland
    30:  "Forest_Vegetation",   # Grassland
    100: "Forest_Vegetation",   # Moss & lichen
    40:  "Cropland",            # Cropland
    80:  "Water",               # Permanent water bodies
    50:  "Built_up",            # Built-up
    60:  "Bare_Sparse",         # Bare / sparse vegetation
    70:  "Bare_Sparse",         # Snow & ice
    90:  "Wetland",             # Herbaceous wetland
    95:  "Wetland",             # Mangroves
}

# ---------------------------------------------------------------------------
# Ordered canonical class list (used for label indices in training)
# ---------------------------------------------------------------------------

GLOBAL6_CLASSES: list[str] = [
    "Forest_Vegetation",
    "Cropland",
    "Water",
    "Built_up",
    "Bare_Sparse",
    "Wetland",
]

# Integer label index → class name (for model output decoding)
GLOBAL6_INDEX: dict[int, str] = {i: cls for i, cls in enumerate(GLOBAL6_CLASSES)}

# Class name → integer label index
GLOBAL6_LABEL: dict[str, int] = {cls: i for i, cls in enumerate(GLOBAL6_CLASSES)}
