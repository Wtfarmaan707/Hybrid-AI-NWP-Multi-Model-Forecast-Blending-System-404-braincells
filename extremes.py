"""
scripts/extremes.py

STEP 5a of the pipeline: Extreme-weather guidance.

Reads:
    data/processed/blended_forecast.csv   (Step 4 output)
    data/processed/history_with_truth.csv (Step 3 input - used only to
                                            derive a rough zone-level
                                            "normal" temperature for the
                                            heat-wave departure check)

Produces:
    data/processed/extreme_events.csv
        One row per (point, rain_day, event_type) that CROSSED a
        threshold - heavy rain, heat wave, or high wind - using the
        BLENDED forecast as the primary signal, plus a model_agreement
        score (what fraction of the 3 raw models also cross the same
        threshold - cheap ensemble-style confidence).

THRESHOLDS (IMD-aligned, documented so the team can defend every number):
    Rainfall (24h accumulated, mm):
        heavy            64.5  - 115.5
        very heavy       115.6 - 204.4
        extremely heavy  > 204.4
    Heat wave (plains/most zones):
        heat wave         daily max temp - zone normal >= 4.5 C,
                           OR absolute daily max >= 45 C
        severe heat wave  daily max temp - zone normal >= 6.4 C,
                           OR absolute daily max >= 47 C
    High wind (10m wind speed, km/h - Open-Meteo's default unit):
        advisory          >= 40
        high wind         >= 62
        severe/damaging   >= 89   (roughly IMD's "squall" threshold)

These are deliberately simple, defensible, IMD-flavoured rules for a
4-day build - not a substitute for IMD's full official criteria (which
also vary by exact region/season). Documented here so anyone can adjust
the numbers in one place (see THRESHOLDS below) without touching logic.

Usage:
    python3 scripts/extremes.py
"""

import os
import numpy as np
import pandas as pd

THRESHOLDS = {
    "rain_heavy": 64.5,
    "rain_very_heavy": 115.6,
    "rain_extreme": 204.4,
    "heat_wave_departure": 4.5,
    "heat_wave_severe_departure": 6.4,
    "heat_wave_absolute": 45.0,
    "heat_wave_severe_absolute": 47.0,
    "wind_advisory": 40.0,
    "wind_high": 62.0,
    "wind_severe": 89.0,
}

# Rough SEPTEMBER daily-max-temperature climatological normals (deg C),
# PER POINT (not per zone) - used only for the heat-wave DEPARTURE check.
# Per-point rather than per-zone because some zones (e.g. "himalayan")
# genuinely mix very different climates - Srinagar/Leh (cold, high
# altitude) sit in the same zone as Itanagar/Shillong (warm NE
# hill-foothill terrain) for the purposes of model-skill grouping, but
# using one zone-wide normal for both was producing false heat-wave
# flags at the warmer stations. These are ballpark real seasonal figures
# (not derived from simulated data). Replace with actual IMD 1981-2010
# station normals before treating departure numbers as authoritative.
POINT_NORMAL_MAX_TEMP_C = {
    "himalayan:srinagar": 28, "himalayan:shimla": 24, "himalayan:dehradun": 31,
    "himalayan:gangtok": 22, "himalayan:shillong": 24, "himalayan:itanagar": 30,
    "himalayan:leh": 20, "himalayan:darjeeling": 20,
    "indo_gangetic:delhi": 35, "indo_gangetic:lucknow": 34, "indo_gangetic:patna": 33,
    "indo_gangetic:amritsar": 34, "indo_gangetic:kanpur": 34, "indo_gangetic:varanasi": 33,
    "indo_gangetic:chandigarh": 33, "indo_gangetic:jaipur": 35, "indo_gangetic:agra": 35,
    "indo_gangetic:bareilly": 33,
    "western_ghats_coast:mumbai": 32, "western_ghats_coast:pune": 30, "western_ghats_coast:goa": 31,
    "western_ghats_coast:mangalore": 31, "western_ghats_coast:kochi": 31,
    "western_ghats_coast:thiruvananthapuram": 31, "western_ghats_coast:kozhikode": 31,
    "western_ghats_coast:ratnagiri": 31, "western_ghats_coast:coorg": 26,
    "deccan_plateau:nagpur": 32, "deccan_plateau:hyderabad": 30, "deccan_plateau:bhopal": 30,
    "deccan_plateau:nashik": 30, "deccan_plateau:raipur": 31, "deccan_plateau:bengaluru": 27,
    "deccan_plateau:indore": 31, "deccan_plateau:aurangabad": 31,
    "east_coast:chennai": 33, "east_coast:visakhapatnam": 32, "east_coast:bhubaneswar": 32,
    "east_coast:kolkata": 32, "east_coast:puducherry": 32, "east_coast:nellore": 33,
    "east_coast:cuttack": 32,
    "peninsular_south:madurai": 34, "peninsular_south:coimbatore": 31,
    "peninsular_south:tiruchirappalli": 34, "peninsular_south:salem": 32,
    "peninsular_south:vellore": 33, "peninsular_south:mysuru": 28,
    "arid_west:jodhpur": 36, "arid_west:bikaner": 37, "arid_west:jaisalmer": 37,
    "arid_west:bhuj": 35, "arid_west:ahmedabad": 35,
}

# Zone-level fallback, used only if a point_id isn't found above (e.g. if
# config/zones.py gets new points added later without updating this table).
SEPTEMBER_NORMAL_MAX_TEMP_C = {
    "himalayan": 22.0,
    "indo_gangetic": 34.0,
    "western_ghats_coast": 31.0,
    "deccan_plateau": 31.0,
    "east_coast": 32.0,
    "peninsular_south": 31.0,
    "arid_west": 36.0,
}


def compute_rain_day(valid_time: pd.Series) -> pd.Series:
    """Same IMD rainfall-day convention used in harmonize.py: 03 UTC ->
    03 UTC. Recomputed here since blend.py doesn't carry rain_day
    through (it operates on hourly data, not daily)."""
    vt = pd.to_datetime(valid_time, utc=True)
    return (vt - pd.Timedelta(hours=3)).dt.date


def point_normals(daily: pd.DataFrame) -> pd.Series:
    """Per-point 'normal' daily-max temperature for the heat-wave
    DEPARTURE check (see POINT_NORMAL_MAX_TEMP_C above for why per-point
    rather than per-zone). Falls back to the zone-level table for any
    point not explicitly listed."""
    zone_fallback = daily["zone"].map(SEPTEMBER_NORMAL_MAX_TEMP_C)
    point_specific = daily["point_id"].map(POINT_NORMAL_MAX_TEMP_C)
    return point_specific.fillna(zone_fallback)


def flag_rain(daily: pd.DataFrame) -> pd.DataFrame:
    v = daily["precip_day_sum_blend"]
    cat = np.select(
        [v > THRESHOLDS["rain_extreme"], v > THRESHOLDS["rain_very_heavy"], v > THRESHOLDS["rain_heavy"]],
        ["extremely_heavy", "very_heavy", "heavy"],
        default="none",
    )
    daily["rain_category"] = cat
    return daily


def flag_heat(daily: pd.DataFrame) -> pd.DataFrame:
    daily["zone_normal_temp"] = point_normals(daily)
    departure = daily["temp_day_max_blend"] - daily["zone_normal_temp"]
    absolute = daily["temp_day_max_blend"]

    is_severe = (departure >= THRESHOLDS["heat_wave_severe_departure"]) | (absolute >= THRESHOLDS["heat_wave_severe_absolute"])
    is_heat = (departure >= THRESHOLDS["heat_wave_departure"]) | (absolute >= THRESHOLDS["heat_wave_absolute"])

    daily["heat_category"] = np.select([is_severe, is_heat], ["severe_heat_wave", "heat_wave"], default="none")
    daily["temp_departure_c"] = departure.round(2)
    return daily


def flag_wind(daily: pd.DataFrame) -> pd.DataFrame:
    v = daily["wind_day_max_blend"]
    cat = np.select(
        [v >= THRESHOLDS["wind_severe"], v >= THRESHOLDS["wind_high"], v >= THRESHOLDS["wind_advisory"]],
        ["severe", "high_wind", "advisory"],
        default="none",
    )
    daily["wind_category"] = cat
    return daily


def model_agreement(daily: pd.DataFrame, model_cols: dict, threshold_col: str, threshold_val: float) -> pd.Series:
    """Fraction of raw models that ALSO cross the given threshold,
    agreeing with the blend's exceedance call. 1.0 = every model agrees
    there's an event here; 0.0 = only the blend thinks so (low trust)."""
    if not model_cols:
        return pd.Series(np.nan, index=daily.index)
    exceed = pd.DataFrame({m: daily[c] >= threshold_val for m, c in model_cols.items()})
    return exceed.mean(axis=1)


def run():
    proc_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    blended_path = os.path.join(proc_dir, "blended_forecast.csv")
    history_path = os.path.join(proc_dir, "history_with_truth.csv")

    print(f"Loading: {blended_path}")
    df = pd.read_csv(blended_path)
    df["rain_day"] = compute_rain_day(df["valid_time"])

    # discover raw per-model columns present for each variable
    def model_cols_for(var):
        cols = [c for c in df.columns if c.startswith(f"{var}_") and not c.endswith(("_blend", "_equal_mean"))]
        return {c[len(var) + 1:]: c for c in cols}

    precip_models = model_cols_for("precipitation")
    temp_models = model_cols_for("temperature_2m")
    wind_models = model_cols_for("wind_speed_10m")

    print("Aggregating to daily (rain-day) level per point...")
    agg_dict = {
        "precip_day_sum_blend": ("precipitation_blend", "sum"),
        "temp_day_max_blend": ("temperature_2m_blend", "max"),
        "wind_day_max_blend": ("wind_speed_10m_blend", "max"),
        "zone": ("zone", "first"),
        "site_name": ("site_name", "first"),
        "lat": ("lat", "first"),
        "lon": ("lon", "first"),
    }
    daily = df.groupby(["point_id", "rain_day"]).agg(**agg_dict).reset_index()

    # per-model daily aggregates, needed for agreement scoring
    for var, models, aggfunc in [
        ("precipitation", precip_models, "sum"),
        ("temperature_2m", temp_models, "max"),
        ("wind_speed_10m", wind_models, "max"),
    ]:
        for model, col in models.items():
            daily_model = df.groupby(["point_id", "rain_day"])[col].agg(aggfunc).reset_index()
            daily_model = daily_model.rename(columns={col: f"{var}_day_{model}"})
            daily = daily.merge(daily_model, on=["point_id", "rain_day"], how="left")

    print("Applying thresholds...")
    daily = flag_rain(daily)
    daily = flag_heat(daily)
    daily = flag_wind(daily)

    precip_model_daycols = {m: f"precipitation_day_{m}" for m in precip_models}
    temp_model_daycols = {m: f"temperature_2m_day_{m}" for m in temp_models}
    wind_model_daycols = {m: f"wind_speed_10m_day_{m}" for m in wind_models}

    daily["rain_model_agreement"] = model_agreement(daily, precip_model_daycols, "rain_category", THRESHOLDS["rain_heavy"])
    daily["wind_model_agreement"] = model_agreement(daily, wind_model_daycols, "wind_category", THRESHOLDS["wind_advisory"])

    events = daily[
        (daily["rain_category"] != "none") | (daily["heat_category"] != "none") | (daily["wind_category"] != "none")
    ].copy()

    out_all_path = os.path.join(proc_dir, "daily_extremes_full.csv")
    daily.to_csv(out_all_path, index=False)
    out_events_path = os.path.join(proc_dir, "extreme_events.csv")
    events.to_csv(out_events_path, index=False)

    print(f"\nWrote full daily table: {len(daily):,} rows -> {out_all_path}")
    print(f"Wrote extreme events only: {len(events):,} rows -> {out_events_path}")

    print("\n--- Event counts by type ---")
    print("Rain:", daily["rain_category"].value_counts().to_string())
    print("\nHeat:", daily["heat_category"].value_counts().to_string())
    print("\nWind:", daily["wind_category"].value_counts().to_string())

    if len(events):
        print("\n--- Sample events ---")
        cols = ["point_id", "zone", "rain_day", "rain_category", "heat_category", "wind_category",
                "precip_day_sum_blend", "temp_day_max_blend", "rain_model_agreement"]
        print(events[cols].head(10).to_string(index=False))

    return out_events_path


if __name__ == "__main__":
    run()
