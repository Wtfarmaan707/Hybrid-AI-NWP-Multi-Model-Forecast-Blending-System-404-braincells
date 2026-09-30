"""
scripts/ingest.py

STEP 1 of the pipeline: Data ingestion.

Pulls forecasts from 3 model families for every representative point in
config/zones.py, using Open-Meteo's free, no-auth Forecast API:

  - gfs_global      -> physical NWP (NOAA)
  - icon_global      -> physical NWP (DWD)
  - ecmwf_aifs025    -> AI/ML model (ECMWF)

Variables pulled (hourly): temperature_2m, precipitation, wind_speed_10m,
wind_gusts_10m, relative_humidity_2m.

Output: one row per (point_id, model, valid_time), written to
data/raw/forecast_<run_date>.csv

This is the ONLY script that touches the network. Everything downstream
(harmonize.py, skill.py, blend.py) reads this file and never calls the
API directly - keeps the rest of the team unblocked by rate limits.

Usage:
    python3 scripts/ingest.py                # pulls today's forecast run
    python3 scripts/ingest.py --test          # pulls just 3 points, fast sanity check
"""

import sys
import os
import time
import argparse
import datetime as dt
import random
import requests
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.zones import all_points

BASE_URL = "https://api.open-meteo.com/v1/forecast"
MODELS = ["gfs_global", "icon_global", "ecmwf_aifs025_single"]
# NOTE: Open-Meteo's general /v1/forecast endpoint requires the "_single" suffix
# for AIFS (ecmwf_aifs025_single). Plain "ecmwf_aifs025" is a valid model id on
# other Open-Meteo endpoints (e.g. ensemble) but returns silent empty columns
# here - not an error, just nulls, which is why this was easy to miss.
HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "relative_humidity_2m",
]
FORECAST_DAYS = 7          # lead time horizon
REQUEST_TIMEOUT = 20
RETRY = 3
SLEEP_BETWEEN_CALLS = 0.3  # be polite to the free API


def fetch_point(point_id: str, zone_key: str, name: str, lat: float, lon: float,
                 run_date: str) -> pd.DataFrame:
    """Fetch all 3 models for one point in a single API call (Open-Meteo
    accepts a comma-separated `models` list and returns one column per
    model per variable)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "models": ",".join(MODELS),
        "forecast_days": FORECAST_DAYS,
        "timezone": "UTC",
    }

    last_err = None
    for attempt in range(RETRY):
        try:
            r = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:
        print(f"  [FAIL] {point_id}: {last_err}")
        return pd.DataFrame()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        print(f"  [EMPTY] {point_id}")
        return pd.DataFrame()

    rows = []
    for model in MODELS:
        for i, t in enumerate(times):
            row = {
                "point_id": point_id,
                "zone": zone_key,
                "site_name": name,
                "lat": lat,
                "lon": lon,
                "model": model,
                "run_date": run_date,
                "valid_time": t,
            }
            for var in HOURLY_VARS:
                key = f"{var}_{model}"          # Open-Meteo's multi-model column naming
                series = hourly.get(key)
                row[var] = series[i] if series and i < len(series) else None
            rows.append(row)

    return pd.DataFrame(rows)


def fetch_point_mock(point_id: str, zone_key: str, name: str, lat: float, lon: float,
                      run_date: str) -> pd.DataFrame:
    """Generate synthetic-but-plausible data with the EXACT same schema as
    fetch_point(). Used when the network is unavailable (e.g. sandboxed dev
    environments) so downstream scripts (harmonize/skill/blend/dashboard)
    can be built and tested without waiting on a live API.

    NOT for the real submission - swap back to fetch_point() before the
    real ingestion run. Each model gets a deliberately different bias/noise
    profile so the skill table in Step 3 has something real to differentiate.
    """
    rng = random.Random(hash(point_id) & 0xffffffff)
    base_time = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    times = [(base_time + dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(FORECAST_DAYS * 24)]

    # crude climatology so zones look distinct even in mock data
    is_coastal = "coast" in zone_key or "ghats" in zone_key
    is_himalaya = zone_key == "himalayan"
    is_arid = zone_key == "arid_west"
    base_temp = 18 if is_himalaya else (26 if is_coastal else (34 if is_arid else 29))
    base_precip_chance = 0.35 if is_coastal else (0.1 if is_arid else 0.2)

    # per-model bias signature (mock stand-in for "real" model skill differences)
    model_profile = {
        "gfs_global":    {"temp_bias": +0.8, "temp_noise": 1.4, "precip_bias": 0.9,  "wind_noise": 1.2},
        "icon_global":   {"temp_bias": -0.3, "temp_noise": 1.0, "precip_bias": 1.15, "wind_noise": 0.9},
        "ecmwf_aifs025": {"temp_bias": +0.1, "temp_noise": 0.7, "precip_bias": 1.0,  "wind_noise": 0.8},
    }

    rows = []
    for model in MODELS:
        prof = model_profile[model]
        for h, t in enumerate(times):
            diurnal = 6 * (1 if 6 <= (h % 24) <= 15 else -1) * abs(((h % 24) - 10) / 10)
            temp = base_temp + diurnal + prof["temp_bias"] + rng.gauss(0, prof["temp_noise"])
            rain = max(0.0, rng.gauss(0, 3)) * prof["precip_bias"] if rng.random() < base_precip_chance else 0.0
            wind = max(0.0, 12 + rng.gauss(0, 4) * prof["wind_noise"])
            gust = wind * (1.4 + rng.random() * 0.3)
            rh = min(100, max(10, 60 + rng.gauss(0, 15)))
            rows.append({
                "point_id": point_id, "zone": zone_key, "site_name": name,
                "lat": lat, "lon": lon, "model": model, "run_date": run_date,
                "valid_time": t,
                "temperature_2m": round(temp, 2),
                "precipitation": round(rain, 2),
                "wind_speed_10m": round(wind, 2),
                "wind_gusts_10m": round(gust, 2),
                "relative_humidity_2m": round(rh, 1),
            })
    return pd.DataFrame(rows)


def run(test_mode: bool = False, mock: bool = False):
    points = all_points()
    if test_mode:
        points = points[:3]
        print(f"[TEST MODE] pulling {len(points)} points only\n")

    run_date = dt.datetime.utcnow().strftime("%Y-%m-%dT%H")
    mode_label = "MOCK" if mock else "LIVE"
    print(f"Ingestion run [{mode_label}]: {run_date} UTC | {len(points)} points x {len(MODELS)} models")

    fetch_fn = fetch_point_mock if mock else fetch_point

    frames = []
    for idx, (point_id, zone_key, name, lat, lon) in enumerate(points, 1):
        print(f"  ({idx}/{len(points)}) {point_id:45s} lat={lat:6.2f} lon={lon:6.2f}")
        df = fetch_fn(point_id, zone_key, name, lat, lon, run_date)
        if not df.empty:
            frames.append(df)
        if not mock:
            time.sleep(SLEEP_BETWEEN_CALLS)

    if not frames:
        print("No data fetched - aborting.")
        return None

    out = pd.concat(frames, ignore_index=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    safe_run = run_date.replace(":", "")
    out_path = os.path.join(out_dir, f"forecast_{safe_run}.csv")
    out.to_csv(out_path, index=False)

    print(f"\nWrote {len(out):,} rows -> {out_path}")
    print(f"Points covered: {out['point_id'].nunique()} / {len(points)}")
    print(f"Models: {sorted(out['model'].unique().tolist())}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="fetch only 3 points for a fast sanity check")
    parser.add_argument("--mock", action="store_true", help="generate synthetic data instead of calling the live API (no network needed)")
    args = parser.parse_args()
    run(test_mode=args.test, mock=args.mock)
