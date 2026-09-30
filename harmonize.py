"""
scripts/harmonize.py

STEP 2 of the pipeline: Harmonization.

Takes the raw ingestion CSV (one row per point x model x hour) and produces
a clean, analysis-ready table:

  - Common time base: valid_time parsed to UTC-aware timestamps; an
    ist_time column added for the dashboard (UTC+5:30), but all internal
    logic stays in UTC.
  - Common rainfall accumulation window: hourly precipitation is rolled up
    into fixed 24h IMD-style rainfall days (03 UTC -> 03 UTC, which is
    08:30 IST -> 08:30 IST, matching IMD's official rainfall day).
  - Derived variables: heat index (simple NWS formula) and a rounded
    lead_hours column (hours since the model run), since skill decays with
    lead time and Step 3 needs this to bucket by lead time.
  - Wide reshape: one row per (point_id, zone, valid_time / rain_day),
    with columns per model, e.g. temperature_2m__gfs_global,
    temperature_2m__icon_global, ... This is the shape Step 3 (skill) and
    Step 4 (blending) both want - models side by side for easy comparison
    and easy weighted averaging.

Handles a variable number of models gracefully - if AIFS is present it's
included automatically; if it's still empty (e.g. mid-fix upstream) it's
dropped with a warning rather than breaking the run.

Usage:
    python3 scripts/harmonize.py <path_to_raw_csv>
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    return df


def drop_empty_models(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any model that is 100% null for its core variables - almost
    always an API/model-id mismatch upstream, not real missing weather.
    Warns loudly so it's never silently swallowed."""
    core_vars = ["temperature_2m", "precipitation"]
    bad_models = []
    for model, g in df.groupby("model"):
        if all(g[v].isna().all() for v in core_vars if v in g.columns):
            bad_models.append(model)
    if bad_models:
        print(f"[WARN] dropping model(s) with 100% null core variables: {bad_models}")
        print("       (usually a model-id mismatch in ingest.py, not a real data gap)")
        df = df[~df["model"].isin(bad_models)].copy()
    return df


def add_lead_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Hours between the model run initialization and the forecast valid
    time. Supports two ingestion schemas:
      - run_date as 'YYYY-MM-DDTHH' (UTC) - original ingest.py format
      - fetch_time_utc as full ISO timestamp, e.g. '2026-09-24T17:47:30Z'

    Open-Meteo always starts the hourly series at 00:00 UTC of the request
    day, regardless of what hour the run was actually issued at - so early
    rows can have a negative lead_hours (they're backfilled/analysis hours
    before the model run, not real forecasts). We flag these with
    is_forecast=False rather than silently dropping them, so Step 3's
    skill-by-lead-time buckets can exclude them cleanly while the raw data
    stays available for anyone who wants to inspect the analysis period."""
    if "run_date" in df.columns:
        run_dt = pd.to_datetime(df["run_date"] + ":00", utc=True, format="%Y-%m-%dT%H:%M")
    elif "fetch_time_utc" in df.columns:
        run_dt = pd.to_datetime(df["fetch_time_utc"], utc=True)
    else:
        raise ValueError("Expected a 'run_date' or 'fetch_time_utc' column in the raw CSV")

    df["lead_hours"] = ((df["valid_time"] - run_dt).dt.total_seconds() / 3600).round().astype(int)
    df["is_forecast"] = df["lead_hours"] >= 0
    n_bad = (~df["is_forecast"]).sum()
    if n_bad:
        print(f"[INFO] {n_bad} rows are pre-run analysis hours (lead_hours < 0), flagged is_forecast=False")
    return df


def add_ist_time(df: pd.DataFrame) -> pd.DataFrame:
    df["ist_time"] = df["valid_time"] + pd.Timedelta(hours=5, minutes=30)
    return df


def add_rain_day(df: pd.DataFrame) -> pd.DataFrame:
    """IMD's official rainfall day runs 03 UTC -> 03 UTC (08:30 IST ->
    08:30 IST next day). Shift valid_time back 3 hours before taking the
    date, so hours 00-02 UTC count toward the PREVIOUS rain day, matching
    IMD convention."""
    shifted = df["valid_time"] - pd.Timedelta(hours=3)
    df["rain_day"] = shifted.dt.date
    return df


def add_heat_index(df: pd.DataFrame) -> pd.DataFrame:
    """Simplified heat index (NWS Rothfusz regression), valid above ~27C.
    Below that it just returns the air temperature - heat index isn't
    meaningful in cooler conditions and the extremes layer only cares
    about the hot end anyway."""
    T = df["temperature_2m"] * 9 / 5 + 32  # to Fahrenheit for the standard formula
    R = df["relative_humidity_2m"]
    hi_f = (
        -42.379 + 2.04901523 * T + 10.14333127 * R
        - 0.22475541 * T * R - 0.00683783 * T**2
        - 0.05481717 * R**2 + 0.00122874 * T**2 * R
        + 0.00085282 * T * R**2 - 0.00000199 * T**2 * R**2
    )
    hi_c = (hi_f - 32) * 5 / 9
    df["heat_index_c"] = np.where(df["temperature_2m"] >= 27, hi_c, df["temperature_2m"])
    return df


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape so each model becomes a column suffix, one row per
    (point_id, valid_time). This is the shape both the skill table and
    the blending engine want."""
    value_cols = ["temperature_2m", "precipitation", "wind_speed_10m",
                  "wind_gusts_10m", "relative_humidity_2m", "heat_index_c"]

    # lead_hours and rain_day/valid_time are per (point, model, run) - since
    # all models share the same run_date and valid_time grid here, we can
    # safely pivot on point_id + valid_time.
    wide = df.pivot_table(
        index=["point_id", "zone", "site_name", "lat", "lon", "valid_time", "ist_time",
               "rain_day", "lead_hours", "is_forecast"],
        columns="model",
        values=value_cols,
        aggfunc="first",
    )
    wide.columns = [f"{var}__{model}" for var, model in wide.columns]
    wide = wide.reset_index()
    return wide


def run(raw_csv_path: str):
    print(f"Loading raw ingestion file: {raw_csv_path}")
    df = load_raw(raw_csv_path)
    print(f"  {len(df):,} rows | {df['point_id'].nunique()} points | models: {sorted(df['model'].unique())}")

    df = drop_empty_models(df)
    df = add_lead_hours(df)
    df = add_ist_time(df)
    df = add_rain_day(df)
    df = add_heat_index(df)

    models_remaining = sorted(df["model"].unique())
    print(f"  models after cleaning: {models_remaining}")

    wide = to_wide(df)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "harmonized.csv")
    wide.to_csv(out_path, index=False)

    print(f"\nWrote {len(wide):,} rows x {len(wide.columns)} cols -> {out_path}")
    print(f"Zones: {sorted(wide['zone'].unique())}")
    print(f"Lead hour range: {wide['lead_hours'].min()} to {wide['lead_hours'].max()}")
    print(f"Rain day range: {wide['rain_day'].min()} to {wide['rain_day'].max()}")
    print("\nSample columns:", [c for c in wide.columns if "temperature" in c])
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", help="path to the raw ingestion CSV from Step 1")
    args = parser.parse_args()
    run(args.raw_csv)
