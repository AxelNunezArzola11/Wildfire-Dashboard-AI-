# Wildfire Agent Run Report

| Field | Value |
|-------|-------|
| Run ID | `750edda188ec4257b1f7991646b227ec` |
| Country | Angola |
| Started (UTC) | 2026-08-15 22:30:15 UTC |
| Guardrail verdict | pass |

---

## Risk Summary

Evidence:
- Active fire count: 8598
- Total FRP: 255203.2 MW
- Maximum FRP: 565.7 MW
- Spread index: 2029964 km²
- High-confidence detections: 28.2%

Confidence: 85%

The current risk level for Angola is EXTREME. There are 8598 active fires with a total fire radiative power of 255203.2 MW. The maximum fire radiative power recorded is 565.7 MW. Detections are spread across a bounding area of 2029964 km². Top hotspots are located at latitudes -8.34, -9.40, -9.48, -10.30, and -8.35 with longitudes 15.95, 22.34, 20.30, 22.08, and 15.95 respectively.

Recommended actions for the next 24 hours:
• Deploy additional aerial surveillance to monitor hotspot areas.
• Increase ground patrols in high-intensity fire zones.
• Coordinate with local authorities to prepare evacuation plans for vulnerable communities.

---

## Forecast Interpretation

Evidence:
- Top 3 of 200 cells shown; 95 at EXTREME, 49 at HIGH, 19 at MEDIUM
- Max fire probability = 92.8% (Cell 1)
- Top cell risk band = EXTREME
- Key driver: low humidity (43.458333333333336), very recent fire activity (24 h) (2.0)

Confidence: 78%

Note: This is a probabilistic estimate based on historical fire activity and weather data. It is not a certainty.

The three highest-risk grid cells are:
1. Lat -9.66, Lon 22.80 – EXTREME risk, 92.8% fire probability. Key drivers: low humidity (43.46%) and very recent fire activity (2.0 incidents in the last 24 hours).
2. Lat -8.41, Lon 18.05 – EXTREME risk, 91.8% fire probability. Key drivers: recent fire history (11.0 incidents in the last 7 days) and very recent fire activity (4.0 incidents in the last 24 hours).
3. Lat -4.92, Lon 22.80 – EXTREME risk, 91.8% fire probability. Key drivers: low humidity (42.625%) and very recent fire activity (2.0 incidents in the last 24 hours).

XGBoost was used for this forecast. As a gradient boosting model, XGBoost provides high predictive accuracy but can be sensitive to overfitting if hyperparameters are not carefully tuned. The model's reliability depends on the quality and representativeness of the training data.

Preparation actions for the next 24 hours:
1. Deploy additional fire watch teams to the three high-risk cells, prioritizing areas with low humidity and recent fire activity.
2. Preposition firefighting resources (water tenders, crews) near the coordinates of the top three cells, especially where recent fire history is high.
3. Issue public alerts advising residents in and around the high-risk areas to remain vigilant, avoid open flames, and prepare evacuation routes.
