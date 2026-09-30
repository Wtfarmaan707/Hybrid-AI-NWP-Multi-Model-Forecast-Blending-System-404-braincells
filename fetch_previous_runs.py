"""
scripts/fetch_previous_runs.py

REAL data puller for Step 3 - use this on your own machine (needs network)
to replace simulate_history.py's synthetic data before the final demo.

Uses Open-Meteo's Previous Runs API, which is built exactly for this: it
returns what each model forecast at a fixed lead-time offset in the past,
which you can directly compare to the model's own most-recent value at
that same timestamp (used here as a stand-in for "truth" - the model's
near-term/analysis value is close to observed reality). For a stronger
final claim, verify against real IMD/observed data instead - this script
is the fast path to SOMETHING real, not a substitute for that.

API: https://previous-runs-api.open-meteo.com/v1/forecast
Docs: https://open-meteo.com/en/docs/previous-runs-api

Output: same schema as simulate_history.py (see history_schema.py), so
skill.py works unmodified on either source.

Usage:
    python3 scripts/fetch_previous_runs.py                 # all 53 points
    python3 scripts/fetch_previous_runs.py --test           # 3 points, fast check
"""

import sys
import os
import time
import argparse
import requests
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.zones import all_points
from scripts.history_schema import REQUIRED_COLUMNS, VARIABLES, LEAD_BUCKETS, validate

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODELS = ["gfs_global", "icon_global", "ecmwf_aifs025_single"]

# Previous Runs API returns one column per (variable, model, lead-day-offset),
# e.g. temperature_2m_gfs_global_previous_day1. We request several fixed
# day-offsets and compare each against the "previous_day0" (i.e. same model's
# most recent/least-lagged estimate) as our truth proxy.
DAY_OFFSETS = [0, 1, 2, 3, 5]  # 0,24,48,72,120 hour leads

HOURLY_VARS = ["temperature_2m", "precipitation", "wind_speed_10m"]
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_CALLS = 0.3


def build_hourly_param():
    params = []
    for var in HOURLY_VARS:
        for model in MODELS:
            for d in DAY_OFFSETS:
                params.append(f"{var}_{model}_previous_day{d}")
    return params


def fetch_point(point_id, zone_key, lat, lon):
    hourly_vars = build_hourly_param()
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "past_days": 14,
        "timezone": "UTC",
    }
    r = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    rows = []
    for var in HOURLY_VARS:
        for model in MODELS:
            truth_key = f"{var}_{model}_previous_day0"
            truth_series = hourly.get(truth_key)
            if not truth_series:
                continue
            for d in DAY_OFFSETS:
                if d == 0:
                    continue  # day0 is the truth proxy itself, not a forecast to verify
                fc_key = f"{var}_{model}_previous_day{d}"
                fc_series = hourly.get(fc_key)
                if not fc_series:
                    continue
                for i, t in enumerate(times):
                    fc_val = fc_series[i] if i < len(fc_series) else None
                    obs_val = truth_series[i] if i < len(truth_series) else None
                    if fc_val is None or obs_val is None:
                        continue
                    rows.append({
                        "point_id": point_id, "zone": zone_key, "model": model,
                        "issue_date": t[:10], "lead_hours": d * 24,
                        "variable": var, "forecast_value": fc_val, "observed_value": obs_val,
                    })
    return pd.DataFrame(rows)


def run(test_mode: bool = False):
    points = all_points()
    if test_mode:
        points = points[:3]
        print(f"[TEST MODE] {len(points)} points\n")

    frames = []
    for idx, (point_id, zone_key, name, lat, lon) in enumerate(points, 1):
        print(f"({idx}/{len(points)}) {point_id}")
        try:
            df = fetch_point(point_id, zone_key, lat, lon)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  [FAIL] {e}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not frames:
        print("No data fetched.")
        return None

    out = pd.concat(frames, ignore_index=True)[REQUIRED_COLUMNS]
    validate(out)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "history_with_truth.csv")
    out.to_csv(out_path, index=False)
    print(f"\n[REAL] Wrote {len(out):,} rows -> {out_path}")
    print("This OVERWRITES the simulated file - skill.py will now use real data.")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run(test_mode=args.test)
