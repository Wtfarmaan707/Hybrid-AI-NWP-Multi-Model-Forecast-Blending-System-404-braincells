"""
scripts/simulate_history.py

Generates a SYNTHETIC (point, model, issue_date, lead_hours, variable)
forecast-vs-truth dataset, matching the schema in history_schema.py.

Why this exists: computing model skill requires comparing PAST forecasts
against what ACTUALLY happened - which today's live ingestion (Step 1)
cannot provide, because nothing in the future has happened yet. Skill
needs history. Until someone runs fetch_previous_runs.py (the real
puller, needs 14+ days of lead time to be worth much), this script lets
Steps 3 (skill table) and 4 (blending) be built and demoed today.

Design choices that make the fake data behave like real weather model
error, so the skill table isn't trivially uniform:
  - Each model has a fixed bias + noise profile (same idea as the mock
    ingestion in Step 1), so some models are systematically better.
  - Error GROWS with lead time (sqrt-ish growth) - this is the single
    most important realistic property, since it's what makes "skill
    depends on lead time" a real, demonstrable finding rather than an
    assumption.
  - Error profile varies by ZONE - coastal/monsoon zones get noisier
    precipitation, matching the real-world reason this project exists.
  - AIFS is deliberately tuned to win on temperature at longer lead
    times (mirrors the real, published finding that AI models often
    hold up better than physical NWP as lead time grows) and to lose
    slightly at very short lead times - so the "different models win in
    different regimes" story is visible in the numbers, not asserted.

REPLACE with fetch_previous_runs.py output before the final demo if you
want to claim real-world verified skill numbers.

Usage:
    python3 scripts/simulate_history.py
    python3 scripts/simulate_history.py --days 21   # more history
"""

import sys
import os
import argparse
import random
import datetime as dt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.zones import all_points, ZONES
from scripts.history_schema import REQUIRED_COLUMNS, VARIABLES, LEAD_BUCKETS, validate

MODELS = ["gfs_global", "icon_global", "ecmwf_aifs025_single"]

# per-model error profile: base_noise (at lead=0) and growth rate per hour
# AIFS is tuned to have a MODEST edge at long lead times (matches the real
# published behaviour of AI models degrading slower than physical NWP),
# but is NOT dominant at short lead - so zone-specific effects (below) can
# actually flip the short/medium-lead winner. This produces a more
# realistic and more demonstrable story: "at short lead, the best model
# depends on terrain; at long lead, AIFS tends to pull ahead everywhere."
MODEL_PROFILE = {
    "gfs_global":            {"temp_bias": 0.5,  "temp_noise0": 0.55, "temp_growth": 0.026,
                               "precip_bias": 1.05, "precip_noise0": 0.8, "precip_growth": 0.05,
                               "wind_bias": 0.4,  "wind_noise0": 0.6, "wind_growth": 0.030},
    "icon_global":            {"temp_bias": -0.3, "temp_noise0": 0.58, "temp_growth": 0.028,
                               "precip_bias": 0.90, "precip_noise0": 0.9, "precip_growth": 0.055,
                               "wind_bias": -0.2, "wind_noise0": 0.7, "wind_growth": 0.028},
    "ecmwf_aifs025_single":   {"temp_bias": 0.15, "temp_noise0": 0.60, "temp_growth": 0.019,
                               "precip_bias": 1.0,  "precip_noise0": 1.1, "precip_growth": 0.035,
                               "wind_bias": 0.1,  "wind_noise0": 0.65,"wind_growth": 0.018},
}

ZONE_PRECIP_NOISE_MULT = {
    "western_ghats_coast": 1.6, "east_coast": 1.5, "himalayan": 1.3,
    "peninsular_south": 1.1, "indo_gangetic": 1.0, "deccan_plateau": 0.9,
    "arid_west": 0.6,
}

# Zone x model temperature noise multipliers, deliberately set to make each
# zone have a CLEAR short-lead winner (mult ~0.6) and two clear laggards
# (mult ~1.25-1.4), so Step 4's weight maps show real regional structure
# instead of one model winning uniformly. Loosely motivated by real
# patterns: ICON (DWD) tends to handle orography/mountain terrain well,
# GFS is often solid over open/arid/well-observed plains, AIFS (AI model)
# tends to generalize best over data-sparse coastal/monsoon regions.
ZONE_TEMP_NOISE_MULT = {
    "himalayan":            {"gfs_global": 1.35, "icon_global": 0.60, "ecmwf_aifs025_single": 1.10},
    "arid_west":             {"gfs_global": 0.60, "icon_global": 1.30, "ecmwf_aifs025_single": 1.15},
    "indo_gangetic":         {"gfs_global": 0.65, "icon_global": 1.20, "ecmwf_aifs025_single": 1.15},
    "western_ghats_coast":   {"gfs_global": 1.30, "icon_global": 1.25, "ecmwf_aifs025_single": 0.60},
    "east_coast":            {"gfs_global": 1.25, "icon_global": 1.20, "ecmwf_aifs025_single": 0.60},
    "peninsular_south":      {"gfs_global": 1.20, "icon_global": 1.15, "ecmwf_aifs025_single": 0.65},
    "deccan_plateau":        {"gfs_global": 0.85, "icon_global": 1.00, "ecmwf_aifs025_single": 0.95},
}


def synth_truth(point_id, zone_key, date, rng):
    """One plausible 'observed' value per variable for a given point/date -
    the thing every model is trying to predict. Simple seasonal/zonal
    climatology, deterministic given the point+date so all models are
    compared against the SAME truth."""
    seed = hash((point_id, str(date))) & 0xffffffff
    r = random.Random(seed)
    is_coastal = "coast" in zone_key or "ghats" in zone_key
    is_himalaya = zone_key == "himalayan"
    is_arid = zone_key == "arid_west"
    base_temp = 18 if is_himalaya else (27 if is_coastal else (35 if is_arid else 29))
    truth_temp = base_temp + r.gauss(0, 2.5)
    precip_chance = 0.4 if is_coastal else (0.1 if is_arid else 0.22)
    truth_precip = max(0.0, r.gauss(8, 6)) if r.random() < precip_chance else 0.0
    truth_wind = max(0.0, 10 + r.gauss(0, 4))
    return truth_temp, truth_precip, truth_wind


def run(days: int = 14):
    points = all_points()
    today = dt.date.today()
    issue_dates = [today - dt.timedelta(days=d) for d in range(1, days + 1)]

    rows = []
    for point_id, zone_key, name, lat, lon in points:
        precip_mult = ZONE_PRECIP_NOISE_MULT.get(zone_key, 1.0)
        temp_mult_by_model = ZONE_TEMP_NOISE_MULT.get(zone_key, {m: 1.0 for m in MODELS})
        for issue_date in issue_dates:
            truth_temp, truth_precip, truth_wind = synth_truth(point_id, zone_key, issue_date, random)
            for lead_h in LEAD_BUCKETS:
                for model in MODELS:
                    prof = MODEL_PROFILE[model]
                    rng = random.Random(hash((point_id, model, str(issue_date), lead_h)) & 0xffffffff)

                    growth = lead_h ** 0.6  # sub-linear growth, realistic shape
                    temp_noise = (prof["temp_noise0"] + prof["temp_growth"] * growth) * temp_mult_by_model.get(model, 1.0)
                    precip_noise = (prof["precip_noise0"] + prof["precip_growth"] * growth) * precip_mult
                    wind_noise = prof["wind_noise0"] + prof["wind_growth"] * growth

                    fc_temp = truth_temp + prof["temp_bias"] + rng.gauss(0, temp_noise)
                    fc_precip = max(0.0, truth_precip * prof["precip_bias"] + rng.gauss(0, precip_noise))
                    fc_wind = max(0.0, truth_wind + prof["wind_bias"] + rng.gauss(0, wind_noise))

                    for var, fc_val, obs_val in [
                        ("temperature_2m", fc_temp, truth_temp),
                        ("precipitation", fc_precip, truth_precip),
                        ("wind_speed_10m", fc_wind, truth_wind),
                    ]:
                        rows.append({
                            "point_id": point_id, "zone": zone_key, "model": model,
                            "issue_date": str(issue_date), "lead_hours": lead_h,
                            "variable": var, "forecast_value": round(fc_val, 2),
                            "observed_value": round(obs_val, 2),
                        })

    df = pd.DataFrame(rows)[REQUIRED_COLUMNS]
    validate(df)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "history_with_truth.csv")
    df.to_csv(out_path, index=False)

    print(f"[SIMULATED] Wrote {len(df):,} rows -> {out_path}")
    print(f"  {len(points)} points x {len(MODELS)} models x {days} days x {len(LEAD_BUCKETS)} lead buckets x {len(VARIABLES)} variables")
    print(f"  This is SYNTHETIC data. Swap in fetch_previous_runs.py output before the final demo.")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14, help="how many past days of history to simulate")
    args = parser.parse_args()
    run(days=args.days)
