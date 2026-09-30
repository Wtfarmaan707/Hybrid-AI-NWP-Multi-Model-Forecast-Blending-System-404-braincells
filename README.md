# Hybrid AI-NWP Forecast Blending — Project Setup

## 1. Open this folder in VS Code

- `File → Open Folder...` → select the unzipped `hybrid_forecast` folder
- Everything below runs from VS Code's built-in terminal: `` Terminal → New Terminal `` (or `` Ctrl+` ``)

## 2. Set up Python

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If VS Code prompts "Select Python Interpreter", pick the one inside `venv/`.

## 3. Folder structure (already set up)

```
hybrid_forecast/
├── config/
│   └── zones.py                  # 53 points across 7 India climate zones
├── scripts/
│   ├── ingest.py                 # Step 1: pull forecasts (GFS, ICON, AIFS)
│   ├── harmonize.py              # Step 2: clean + zone-tag + reshape
│   ├── history_schema.py         # shared schema for skill-table inputs
│   ├── simulate_history.py       # Step 3a: synthetic truth (fast, no network)
│   ├── fetch_previous_runs.py    # Step 3a: REAL truth (needs network)
│   └── skill.py                  # Step 3b: compute the skill table
└── data/
    ├── raw/                      # ingest.py output lands here
    └── processed/                # harmonize.py / skill.py output lands here
```

## 4. Run the pipeline, in order

```bash
# Step 1 — pull real forecasts (needs internet)
python3 scripts/ingest.py --test     # 3-point sanity check first
python3 scripts/ingest.py            # full 53-point pull

# Step 2 — harmonize (point it at whatever Step 1 just wrote)
python3 scripts/harmonize.py data/raw/forecast_<timestamp>.csv

# Step 3 — skill table
python3 scripts/simulate_history.py --days 14      # fast synthetic version
# OR, when you have time / want real numbers for the final demo:
python3 scripts/fetch_previous_runs.py

python3 scripts/skill.py
```

Every script prints exactly what it wrote and where — check the terminal output after each run.

## 5. Already-included sample data

`data/raw/` and `data/processed/` already contain output from a completed
run (real 3-model forecast pull + simulated skill table), so you can jump
straight to Step 4 (blending) without re-running Steps 1-3 if you just
want to keep building.

## 6. Known things to be aware of

- **AIFS model id**: Open-Meteo's general forecast endpoint needs
  `ecmwf_aifs025_single`, not `ecmwf_aifs025` — already fixed in `ingest.py`.
- **AIFS has no `wind_gusts_10m`** on this endpoint — expected, not a bug;
  that column is null for AIFS specifically.
- **`is_forecast` flag** in `harmonized.csv`: Open-Meteo always starts its
  hourly series at 00:00 UTC of the request day, so some early rows are
  pre-run "analysis" hours, not real forecasts. Filter to `is_forecast=True`
  for anything skill/lead-time related.
- **`skill_table.csv` is currently built from simulated truth data**
  (clearly marked in `simulate_history.py`). Swap in
  `fetch_previous_runs.py` output before the final demo for real numbers —
  no downstream code changes needed, the schema is identical.

## 7. Next step

Step 4 (blending engine): inverse-error weighting per zone/lead time,
using `data/processed/skill_table.csv` + `data/processed/harmonized.csv`
to produce the final blended forecast and the model weight map.
