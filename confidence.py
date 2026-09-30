"""
scripts/confidence.py

STEP 5b of the pipeline: Disagreement / Confidence Index (novelty layer).

Core idea: when the 3 models agree closely, that's a genuinely different
situation than when they diverge sharply - and that disagreement is
itself a predictive signal, not just noise to average away. This script
turns per-point, per-hour cross-model spread into a per-zone, per-day
Forecast Confidence Index (green/amber/red), the kind of thing a
forecaster or a farmer/dam-operator dashboard user can act on directly.

Reads:
    data/processed/blended_forecast.csv   (Step 4 output - has each raw
                                            model's value plus the blend)

Produces:
    data/processed/confidence_index.csv
        One row per (point, valid_time): spread (max-min across models),
        a normalized 0-100 confidence score, and a confidence_level
        (green/amber/red) - for temperature, precipitation, and wind.

    data/processed/zone_confidence_daily.csv
        Aggregated to (zone, rain_day): mean confidence per variable,
        for the dashboard's map view and for the "confidence vs lead
        time" chart.

    data/processed/confidence_map_<lead>h.svg
        A quick visual: India colored green/amber/red by confidence at a
        chosen lead time - the literal "novelty" deliverable.

METHOD:
    For each variable, spread = max(model values) - min(model values) at
    that point/hour. Spread is converted to a 0-100 confidence score by
    comparing it against that variable's own spread distribution (a
    percentile-based normalization, not an arbitrary fixed cutoff) - so
    "high disagreement" means high RELATIVE to how much these models
    normally disagree for that variable, not an arbitrary absolute
    number pulled out of thin air.

    confidence = 100 * (1 - percentile_rank(spread))

    green  : confidence >= 70  (models agree closely - trust the blend)
    amber  : 40 <= confidence < 70
    red    : confidence < 40   (models diverge sharply - flag for a human)

Usage:
    python3 scripts/confidence.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.zones import ZONES

VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m"]

GREEN_THRESHOLD = 70
AMBER_THRESHOLD = 40


def compute_spread(df: pd.DataFrame, var: str) -> pd.Series:
    model_cols = [c for c in df.columns if c.startswith(f"{var}_") and not c.endswith(("_blend", "_equal_mean"))]
    if not model_cols:
        return pd.Series(np.nan, index=df.index)
    vals = df[model_cols]
    return vals.max(axis=1) - vals.min(axis=1)


def spread_to_confidence(spread: pd.Series) -> pd.Series:
    """Percentile-rank based: a spread at the 10th percentile (low,
    relative to this variable's own typical disagreement) gets high
    confidence; a spread at the 90th percentile gets low confidence."""
    pct = spread.rank(pct=True, na_option="keep")
    return ((1 - pct) * 100).round(1)


def level_from_confidence(conf: pd.Series) -> pd.Series:
    return np.select(
        [conf >= GREEN_THRESHOLD, conf >= AMBER_THRESHOLD],
        ["green", "amber"],
        default="red",
    )


def compute_point_confidence(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["point_id", "zone", "site_name", "lat", "lon", "valid_time", "lead_hours", "lead_bucket"]].copy()
    for var in VARIABLES:
        spread = compute_spread(df, var)
        conf = spread_to_confidence(spread)
        out[f"{var}_spread"] = spread
        out[f"{var}_confidence"] = conf
        out[f"{var}_confidence_level"] = level_from_confidence(conf.fillna(0))

    conf_cols = [f"{v}_confidence" for v in VARIABLES]
    out["overall_confidence"] = out[conf_cols].mean(axis=1).round(1)
    out["overall_level"] = level_from_confidence(out["overall_confidence"].fillna(0))
    return out


def compute_zone_daily(point_conf: pd.DataFrame) -> pd.DataFrame:
    df = point_conf.copy()
    vt = pd.to_datetime(df["valid_time"], utc=True)
    df["rain_day"] = (vt - pd.Timedelta(hours=3)).dt.date

    agg = {f"{v}_confidence": "mean" for v in VARIABLES}
    agg["overall_confidence"] = "mean"

    zone_daily = df.groupby(["zone", "rain_day"]).agg(agg).reset_index()
    zone_daily = zone_daily.round(1)
    zone_daily["overall_level"] = level_from_confidence(zone_daily["overall_confidence"])
    return zone_daily


def render_confidence_map(zone_daily: pd.DataFrame, lead_hours_label: str = "all") -> str:
    level_color = {"green": "#2fa66a", "amber": "#d9a441", "red": "#c15a3c"}

    # one representative day (the first available) per zone for a clean single map
    first_day = zone_daily["rain_day"].min()
    snap = zone_daily[zone_daily["rain_day"] == first_day].set_index("zone")

    def zone_centroid(zone_key):
        pts = ZONES[zone_key]["points"]
        lat = sum(p[1] for p in pts) / len(pts)
        lon = sum(p[2] for p in pts) / len(pts)
        return lat, lon

    def project(lat, lon):
        x = (lon - 68) / (98 - 68) * 640 + 40
        y = (38 - lat) / (38 - 6) * 520 + 30
        return x, y

    shapes = []
    for zone_key in ZONES:
        if zone_key not in snap.index:
            continue
        row = snap.loc[zone_key]
        lat, lon = zone_centroid(zone_key)
        x, y = project(lat, lon)
        color = level_color.get(row["overall_level"], "#888")
        conf = row["overall_confidence"]
        shapes.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="42" fill="{color}" fill-opacity="0.75" stroke="#111" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{y+4:.1f}" font-family="sans-serif" font-size="13" font-weight="bold" '
            f'text-anchor="middle" fill="#0d1013">{conf:.0f}</text>'
            f'<text x="{x:.1f}" y="{y+55:.1f}" font-family="sans-serif" font-size="11" '
            f'text-anchor="middle" fill="#333">{ZONES[zone_key]["label"]}</text>'
        )

    legend = ""
    ly = 40
    for level, color in level_color.items():
        legend += f'<circle cx="700" cy="{ly}" r="8" fill="{color}"/><text x="716" y="{ly+4}" font-family="sans-serif" font-size="13" fill="#222">{level}</text>'
        ly += 26

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 570">
<rect width="820" height="570" fill="#f5f6f2"/>
<text x="40" y="24" font-family="sans-serif" font-size="18" font-weight="bold" fill="#10202e">Forecast Confidence Index — {first_day}</text>
{''.join(shapes)}
{legend}
<text x="40" y="555" font-family="sans-serif" font-size="11" fill="#5b6c78">Number = overall confidence (0-100). Green = models agree, trust the blend. Red = models diverge, flag for a human.</text>
</svg>'''
    return svg


def run():
    proc_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    blended_path = os.path.join(proc_dir, "blended_forecast.csv")

    print(f"Loading: {blended_path}")
    df = pd.read_csv(blended_path)
    df = df[df["lead_hours"] >= 0].copy()  # only real forecast hours
    print(f"  {len(df):,} forecast rows")

    point_conf = compute_point_confidence(df)
    point_path = os.path.join(proc_dir, "confidence_index.csv")
    point_conf.to_csv(point_path, index=False)
    print(f"Wrote point-level confidence: {len(point_conf):,} rows -> {point_path}")

    zone_daily = compute_zone_daily(point_conf)
    zone_path = os.path.join(proc_dir, "zone_confidence_daily.csv")
    zone_daily.to_csv(zone_path, index=False)
    print(f"Wrote zone-daily confidence: {len(zone_daily):,} rows -> {zone_path}")

    svg = render_confidence_map(zone_daily)
    svg_path = os.path.join(proc_dir, "confidence_map.svg")
    with open(svg_path, "w") as f:
        f.write(svg)
    print(f"Wrote confidence map -> {svg_path}")

    print("\n--- Overall confidence level counts ---")
    print(point_conf["overall_level"].value_counts().to_string())

    print("\n--- Zone-level confidence, by day (overall) ---")
    print(zone_daily.pivot_table(index="zone", columns="rain_day", values="overall_confidence").round(0).to_string())

    print("\n--- Does confidence correlate with lead time? (temperature) ---")
    by_lead = point_conf.groupby("lead_bucket")["temperature_2m_confidence"].mean().round(1)
    print(by_lead.to_string())

    return point_path, zone_path, svg_path


if __name__ == "__main__":
    run()
