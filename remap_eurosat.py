"""
remap_eurosat.py — Remap existing on-disk EuroSAT dataset to the Global-6
land-cover scheme defined in landcover_schema.py.

No re-download: scans data/eurosat/2750/ (or the path you specify) and
builds a label map entirely from directory names.

Usage
-----
    python3 remap_eurosat.py                        # uses default data/eurosat/2750
    python3 remap_eurosat.py data/eurosat/2750       # explicit path

Output
------
    Prints a class-distribution table (old EuroSAT classes → new Global-6 class)
    and the final per-Global-6-class sample counts.
    Writes data/eurosat/global6_labels.csv  (filepath, original_class, global6_class)
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

from landcover_schema import (
    EUROSAT_TO_GLOBAL6,
    GLOBAL6_CLASSES,
    GLOBAL6_LABEL,
)

_DEFAULT_EUROSAT_DIR = Path("data/eurosat/2750")


def remap_eurosat(eurosat_dir: Path) -> list[dict]:
    """
    Walk eurosat_dir, find all image files, and assign Global-6 labels.

    Returns a list of dicts: {filepath, original_class, global6_class, global6_index}
    Raises AssertionError if any directory name is missing from EUROSAT_TO_GLOBAL6.
    """
    records = []
    found_classes = set()

    for class_dir in sorted(eurosat_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        original_class = class_dir.name
        found_classes.add(original_class)

        # Validate mapping before processing any files
        assert original_class in EUROSAT_TO_GLOBAL6, (
            f"EuroSAT class '{original_class}' has no entry in EUROSAT_TO_GLOBAL6. "
            f"Add it to landcover_schema.py before proceeding."
        )
        global6_class = EUROSAT_TO_GLOBAL6[original_class]

        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                records.append({
                    "filepath": str(img_file),
                    "original_class": original_class,
                    "global6_class": global6_class,
                    "global6_index": GLOBAL6_LABEL[global6_class],
                })

    # Confirm every key in EUROSAT_TO_GLOBAL6 that exists on disk is covered
    for cls in found_classes:
        assert cls in EUROSAT_TO_GLOBAL6, f"Unmapped class on disk: '{cls}'"

    print(f"\n  EuroSAT classes found on disk: {sorted(found_classes)}")
    missing_from_disk = set(EUROSAT_TO_GLOBAL6.keys()) - found_classes
    if missing_from_disk:
        print(f"  Note: these EUROSAT_TO_GLOBAL6 keys have no on-disk folder: "
              f"{sorted(missing_from_disk)}")
    print(f"  All {len(found_classes)} on-disk classes successfully mapped.")
    return records


def print_distribution(records: list[dict]):
    """Print old-class breakdown and new Global-6 totals."""
    # Per original class
    per_orig: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in records:
        per_orig[r["original_class"]][r["global6_class"]] += 1

    print("\n  EuroSAT original class  →  Global-6 class          Samples")
    print("  " + "-" * 62)
    for orig_cls in sorted(per_orig.keys()):
        for g6_cls, count in sorted(per_orig[orig_cls].items()):
            print(f"  {orig_cls:<26}→  {g6_cls:<24} {count:>7,}")

    # Per Global-6 class
    per_g6: dict[str, int] = defaultdict(int)
    for r in records:
        per_g6[r["global6_class"]] += 1

    total = sum(per_g6.values())
    print(f"\n  Global-6 class distribution after remapping:")
    print(f"  {'Class':<24}  {'Samples':>8}  {'%':>6}")
    print("  " + "-" * 44)
    for cls in GLOBAL6_CLASSES:
        count = per_g6.get(cls, 0)
        pct = 100.0 * count / total if total else 0
        print(f"  {cls:<24}  {count:>8,}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<24}  {total:>8,}  100.0%")

    # Imbalance note
    max_cls = max(per_g6, key=per_g6.get)
    min_cls = min(per_g6, key=per_g6.get)
    ratio = per_g6[max_cls] / max(per_g6[min_cls], 1)
    print(f"\n  Imbalance ratio (max/min): {per_g6[max_cls]:,} ({max_cls})"
          f" / {per_g6[min_cls]:,} ({min_cls}) = {ratio:.1f}×")
    print("  Note: imbalance is expected and will be handled at training time "
          "(weighted loss / oversampling). No fix needed in this task.")


def save_csv(records: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "original_class", "global6_class", "global6_index"]
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  Saved label CSV: {output_path}  ({len(records):,} rows)")


if __name__ == "__main__":
    eurosat_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_EUROSAT_DIR

    if not eurosat_dir.exists():
        print(f"ERROR: EuroSAT directory not found: {eurosat_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  EuroSAT → Global-6 Remapping")
    print(f"  Source: {eurosat_dir.resolve()}")
    print(f"{'='*60}")

    records = remap_eurosat(eurosat_dir)
    print_distribution(records)

    output_csv = eurosat_dir.parent / "global6_labels.csv"
    save_csv(records, output_csv)

    print(f"\n  Done. {len(records):,} samples remapped.")
