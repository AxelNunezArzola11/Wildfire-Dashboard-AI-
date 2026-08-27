"""
diagnose_shap_features.py

Prints the raw weather feature values for the top-10 highest-risk grid cells
and diagnoses whether identical SHAP outputs are a data issue or a code bug.

Run with:
    python diagnose_shap_features.py [COUNTRY]
"""

import sys
import numpy as np
import pandas as pd

import config
import forecast_engine as fe
from forecast_engine import (
    _build_grid, _get_fire_window, _build_feature_matrix,
    _get_model_and_predictions, _compute_shap_contribs,
    MODEL_FEATURE_COLS, _FEATURE_LABELS,
)


def main():
    country = sys.argv[1] if len(sys.argv) > 1 else next(iter(config.COUNTRY_BBOX))
    print(f"\n=== Diagnosing feature values for: {country} ===\n")

    bbox_str = config.COUNTRY_BBOX.get(country)
    if not bbox_str:
        print(f"ERROR: No bbox for '{country}'")
        return

    grid_points = _build_grid(bbox_str)
    print(f"Grid: {len(grid_points)} points, step = {config.FORECAST_GRID_DEG}° "
          f"(~{config.FORECAST_GRID_DEG * 111:.0f} km)")
    print(f"Open-Meteo native resolution: 0.1° (~11 km)")
    print(f"Grid spacing vs Open-Meteo resolution: "
          f"{config.FORECAST_GRID_DEG / 0.1:.1f}× larger than API native grid\n")

    # Fetch fire windows
    fire_7d_df = _get_fire_window(country, min_frp=0.0, days=7)
    fire_short_df = fire_7d_df.head(0)

    # Build the full feature matrix
    feature_df = _build_feature_matrix(grid_points, fire_7d_df, fire_short_df)
    probabilities, model_used, clf, X_all = _get_model_and_predictions(feature_df)
    feature_df = feature_df.reset_index(drop=True)

    print(f"Model used: {model_used}\n")

    # Build cells list same as run_forecast
    cells_idx = []
    for i, row in feature_df.iterrows():
        prob = float(np.clip(probabilities[i], 0.0, 1.0))
        cells_idx.append((prob, i, float(row["lat"]), float(row["lon"])))
    cells_idx.sort(key=lambda x: x[0], reverse=True)

    top10 = cells_idx[:10]

    WEATHER_FEATS = MODEL_FEATURE_COLS

    print(f"{'Rank':<5} {'Lat':>8} {'Lon':>8} {'Prob%':>6}  "
          + "  ".join(f"{f[:14]:>14}" for f in WEATHER_FEATS))
    print("-" * 120)

    for rank, (prob, row_idx, lat, lon) in enumerate(top10, 1):
        row = feature_df.loc[row_idx]
        vals = [row[f] for f in WEATHER_FEATS]
        val_str = "  ".join(f"{v:>14.4f}" for v in vals)
        print(f"{rank:<5} {lat:>8.4f} {lon:>8.4f} {prob*100:>5.1f}%  {val_str}")

    print("\n── Uniqueness check among top-10 feature rows ──")
    top10_indices = [idx for _, idx, _, _ in top10]
    feat_rows = feature_df.loc[top10_indices][WEATHER_FEATS]
    dupe = feat_rows.duplicated().sum()
    print(f"  Fully duplicate rows among top-10: {dupe} / {len(top10)}")
    for col in WEATHER_FEATS:
        n_unique = feat_rows[col].nunique()
        status = "✅ varies" if n_unique > 1 else "⚠️  CONSTANT"
        print(f"    {col:25s}: {n_unique:>2} unique value(s)  {status}")

    if model_used == "XGBoost" and clf is not None and X_all is not None:
        print("\n── SHAP computation diagnostic ──")
        # Replicate the row-lookup from run_forecast
        top10_row_indices = []
        for (_, _, lat, lon) in top10:
            idx_match = feature_df.index[
                (feature_df["lat"].round(4) == round(lat, 4)) &
                (feature_df["lon"].round(4) == round(lon, 4))
            ].tolist()
            if idx_match:
                top10_row_indices.append(idx_match[0])
            else:
                print(f"  WARNING: No match for lat={lat}, lon={lon}")

        print(f"  top10_row_indices = {top10_row_indices}")

        # Check for duplicate indices (this would mean same row fed to SHAP twice)
        n_unique_indices = len(set(top10_row_indices))
        if n_unique_indices < len(top10_row_indices):
            print(f"  ⚠️  BUG DETECTED: {len(top10_row_indices) - n_unique_indices} "
                  f"duplicate row index(es) — same feature row mapped to multiple cells!")
        else:
            print(f"  ✅ All {n_unique_indices} row indices are unique")

        shap_lists = _compute_shap_contribs(clf, X_all, top10_row_indices)

        print("\n── SHAP values per cell (top-10) ──")
        for rank, (contribs, (prob, row_idx, lat, lon)) in enumerate(
            zip(shap_lists, top10), 1
        ):
            vals_str = "  ".join(
                f"{c['label']}: {c['shap']:+.4f} ({c['pct']:+.1f}%)"
                for c in contribs
            )
            print(f"  Cell {rank:>2} ({lat:.4f},{lon:.4f}) prob={prob*100:.1f}%:  {vals_str}")

        print("\n── Are SHAP outputs identical across cells? ──")
        shap_vecs = [tuple(c["shap"] for c in contribs) for contribs in shap_lists]
        n_unique_shap = len(set(shap_vecs))
        if n_unique_shap == 1:
            print("  ⚠️  ALL CELLS HAVE IDENTICAL SHAP VALUES — potential bug or flat features")
        else:
            print(f"  ✅ {n_unique_shap} distinct SHAP vectors across {len(shap_vecs)} cells")
    else:
        print(f"\nℹ️  Skipping SHAP test — model_used='{model_used}' "
              "(no XGBoost clf available)")
        print("   This means the UI will show the deterministic fallback caption, "
              "not SHAP bars — so any identical-SHAP report must come from a session "
              "where XGBoost DID run.")
        print("\n── SYNTHETIC SHAP BUG TEST (verifying loop logic with fake clf) ──")
        print("   Building a minimal XGBoost on the same feature_df to test "
              "whether _compute_shap_contribs correctly produces per-row outputs...")

        from xgboost import XGBClassifier
        import shap as _shap

        # Make fake labels — split by median probability
        fake_probs = np.array([_fe_det_score(row) for _, row in feature_df.iterrows()])
        median_p = np.median(fake_probs)
        fake_labels = (fake_probs >= median_p).astype(int)

        # Need both classes
        if len(np.unique(fake_labels)) < 2:
            print("   Cannot test — only one class in synthetic labels, skipping.")
            return

        X_all_syn = feature_df[MODEL_FEATURE_COLS].values
        clf_syn = XGBClassifier(n_estimators=10, max_depth=2, random_state=42,
                                eval_metric="logloss", verbosity=0)
        clf_syn.fit(X_all_syn, fake_labels)

        syn_shap = _compute_shap_contribs(clf_syn, X_all_syn, list(range(min(10, len(X_all_syn)))))
        syn_vecs = [tuple(c["shap"] for c in contribs) for contribs in syn_shap]
        n_unique_syn = len(set(syn_vecs))
        if n_unique_syn == 1:
            print("   ⚠️  SYNTHETIC TEST SHOWS FLAT SHAP — the _compute_shap_contribs loop is broken")
        else:
            print(f"   ✅ {n_unique_syn} distinct synthetic SHAP vectors — loop logic is correct")
            print("   → Identical SHAP in the UI is due to identical input features, not a code bug")


def _fe_det_score(row):
    """Local copy of deterministic score to avoid import side-effects."""
    import math
    def safe(val):
        try:
            v = float(val)
            return 0.0 if math.isnan(v) else v
        except (TypeError, ValueError):
            return 0.0
    score = (
        0.30 * min(safe(row.hist_fire_count_7d) / 10.0, 1.0)
        + 0.25 * max(1.0 - safe(row.humidity_24h_mean) / 100.0, 0.0)
        + 0.20 * min(max(safe(row.temp_24h_mean) - 20.0, 0.0) / 30.0, 1.0)
        + 0.15 * min(safe(row.wind_24h_max) / 60.0, 1.0)
        + 0.10 * max(1.0 - safe(row.precip_24h_sum) / 10.0, 0.0)
    )
    import numpy as np_
    return float(np_.clip(score, 0.0, 1.0))


if __name__ == "__main__":
    main()
