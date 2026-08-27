# Wildfire Agent Run Report

| Field | Value |
|-------|-------|
| Run ID | `2170856dfaf64f0f9d43d2f62008f997` |
| Country | Brazil |
| Started (UTC) | 2026-08-15 23:13:34 UTC |
| Guardrail verdict | pass |
| Model used | XGBoost |
| Risk-metrics window | Last 2 days (48h) of FIRMS detections |
| Forecast training window | Last 7 days of FIRMS detections |

✅ XGBoost booster trained and saved to `model.json`.

## Artifact inventory

| File | Role |
|------|------|
| `dataset.csv.gz` | 2-day (48h) FIRMS detections — risk-metrics window only |
| `dataset_forecast_window.csv.gz` | 7-day FIRMS window — **actual XGBoost training input** (9,514 rows) |
| `model.json` | Trained XGBoost booster (native JSON format) |
| `model_script.py` | Standalone training script (wildfire_model_export.py) |
| `report.md` | This file |

---

## Risk Summary

Evidence:
- Active fire count: 3689
- Total FRP: 127981.9 MW
- Maximum FRP: 830.9 MW
- Spread index: 18562905 km²
- High-confidence detections: 23.5%

Confidence: 85%

The current risk level for Brazil is EXTREME. There are 3689 active fires with a total fire radiative power (FRP) of 127981.9 MW, and the maximum FRP recorded is 830.9 MW. Detections are spread across a bounding area of 18562905 km². The top hotspots are located at latitudes -6.97, -6.97, -12.14, -6.71, and -20.74 with longitudes -59.29, -59.29, -47.16, -58.94, and -61.17 respectively.

Recommended actions for the next 24 hours:
• Deploy additional aerial and ground resources to the top hotspot areas.
• Increase public awareness campaigns on fire prevention and safety measures.
• Coordinate with neighboring regions to share resources and intelligence.

---

## Forecast Interpretation

Evidence:
- Top 3 of 200 cells shown; 0 at EXTREME, 0 at HIGH, 79 at MEDIUM
- Max fire probability = 47.6% (Cell 1, Cell 2, Cell 3)
- Top cell risk band = MEDIUM
- Key drivers: low humidity (44.125–44.791666666666664), high wind speed (21.9–30.0)

Confidence: 68%

Note: This is a probabilistic estimate based on historical fire activity and weather data. It is not a certainty.

The three highest-risk grid cells are:
1. Lat -23.12, Lon -59.12 (MEDIUM, 47.6%): Low humidity (44.125) and high wind speed (25.5)
2. Lat -17.62, Lon -61.37 (MEDIUM, 47.6%): Low humidity (45.625) and high wind speed (30.0)
3. Lat -17.12, Lon -45.37 (MEDIUM, 47.6%): Low humidity (44.791666666666664) and high wind speed (21.9)

XGBoost was used, providing a robust, tree-based model that captures complex interactions between weather variables and fire risk, though with moderate interpretability.

Preparation actions for the next 24 hours:
1. Increase patrols and monitoring in the identified grid cells, especially during peak wind periods.
2. Issue public advisories urging caution with open flames and equipment use in high-risk areas.
3. Coordinate with local fire agencies to pre-position resources and readiness teams in the affected regions.
