"""
country_bboxes.py — Shared bounding-box registry for wildfire-covered countries.

Intentionally dependency-free: no imports, no os.environ reads, no credentials.
This module exists so both config.py (the Streamlit app) and
wildfire_model_export.py (the standalone script) can import the single
authoritative copy instead of each maintaining a hardcoded duplicate.

Format: "W,S,E,N" decimal degrees (WGS-84).
"""

COUNTRY_BBOX: dict[str, str] = {
    "Brazil":                        "-73.99,-33.75,-28.85,5.27",
    "Australia":                     "113.34,-43.64,153.57,-10.68",
    "United States":                 "-124.74,24.52,-66.95,49.38",
    "Canada":                        "-141.00,41.68,-52.62,83.11",
    "Indonesia":                     "95.01,-11.01,141.02,5.91",
    "Russia":                        "19.64,41.19,180.00,81.86",
    "Democratic Republic of Congo":  "12.18,-13.46,31.30,5.39",
    "Angola":                        "11.67,-18.04,24.08,-4.39",
    "Mozambique":                    "30.21,-26.87,40.84,-10.47",
    "Mexico":                        "-117.13,14.53,-86.70,32.72",
    "Bolivia":                       "-69.65,-22.90,-57.47,-9.67",
    "Venezuela":                     "-73.35,0.65,-59.80,12.20",
    "Argentina":                     "-73.56,-55.06,-53.64,-21.78",
    "India":                         "68.18,8.07,97.40,35.51",
    "China":                         "73.50,18.16,134.77,53.56",
    "Nigeria":                       "2.69,4.27,14.68,13.89",
    "South Africa":                  "16.46,-34.83,32.89,-22.13",
    "Portugal":                      "-9.53,36.96,-6.19,42.15",
    "Greece":                        "19.37,34.80,29.64,41.75",
    "Chile":                         "-75.64,-55.90,-66.96,-17.51",
}
