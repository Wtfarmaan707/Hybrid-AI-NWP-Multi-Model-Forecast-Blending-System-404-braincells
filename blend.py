"""
scripts/blend.py

STEP 4 of the pipeline: Blending engine.

Reads:
    data/processed/skill_table.csv   (Step 3 - who's good where/when)
    data/processed/harmonized.csv    (Step 2 - the actual forecasts to blend)

Produces THREE artifacts:
    1. data/processed/weights_table.csv
       One row per (zone, lead_bucket, variable, model) -> weight [0,1].
       This IS the "model weight map" deliverable - literally which model
       is trusted where, for every region/lead-time/variable combination.

    2. data/processed/blended_forecast.csv
       One row per (point, valid_time), with:
         - <var>_blend       : Level 1 inverse-error-weighted blend
         - <var>_equal_mean  : Level 0 equal-weight baseline (for comparison)
         - <var>_<model>     : each model's original raw value (kept for
                                the dashboard / for computing spread later)

    3. Console summary showing how much the weights actually vary by
       region - the evidence that this isn't just an equal-weight average
       wearing a fancier name.

METHOD (Level 1 - inverse error weighting):
    w_m = (1 / rmse_m^alpha) / sum_k (1 / rmse_k^alpha)
    computed independently per (zone, lead_bucket, variable), using RMSE
    from the skill table. alpha controls how aggressively skill differences
    get rewarded: alpha=1 is gentle, alpha=2 is sharp. Default alpha=1.5.

Each row in harmonized.csv has a continuous lead_hours value (since actual
forecast hours don't land exactly on the skill table's 9 fixed buckets);
we snap to the NEAREST bucket for the weight lookup. This is a deliberate
simplification - a production system would interpolate between buckets,
but nearest-bucket keeps the logic auditable for a 4-day build.

Usage:
    python3 scripts/blend.py
    python3 scripts/blend.py --alpha 2.0
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.history_schema import LEAD_BUCKETS

VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m"]


def compute_weights(skill: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """w_m = (1/rmse_m^alpha) / sum_k(1/rmse_k^alpha), per (zone, lead_hours, variable).
    A tiny epsilon guards against a zero-RMSE bucket (shouldn't happen with
    real error, but simulated/thin data can occasionally produce one)."""
    df = skill.copy()
    eps = 1e-6
    df["inv_err"] = 1.0 / (df["rmse"] + eps) ** alpha

    df["weight"] = df.groupby(["zone", "lead_hours", "variable"])["inv_err"] \
                      .transform(lambda s: s / s.sum())

    return df[["zone", "lead_hours", "variable", "model", "weight", "rmse", "n"]]


def snap_to_bucket(lead_hours: pd.Series) -> pd.Series:
    """Map each row's actual lead_hours to the nearest of the 9 fixed
    LEAD_BUCKETS used throughout the skill table."""
    buckets = np.array(LEAD_BUCKETS)
    lh = lead_hours.to_numpy()
    idx = np.abs(lh[:, None] - buckets[None, :]).argmin(axis=1)
    return pd.Series(buckets[idx], index=lead_hours.index)


def blend_forecast(harmonized: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    df = harmonized.copy()
    df = df[df["is_forecast"]].copy()   # never blend pre-run analysis rows
    df["lead_bucket"] = snap_to_bucket(df["lead_hours"])

    # discover which models are actually present as columns, per variable
    model_cols = {}
    for var in VARIABLES:
        cols = [c for c in df.columns if c.startswith(f"{var}__")]
        model_cols[var] = {c.split("__", 1)[1]: c for c in cols}

    out_frames = []
    for var in VARIABLES:
        models_here = model_cols[var]
        if not models_here:
            continue

        w_var = weights[weights["variable"] == var]
        # pivot to zone x lead_bucket x model -> weight, for a fast merge
        w_wide = w_var.pivot_table(index=["zone", "lead_hours"], columns="model", values="weight").reset_index()
        w_wide = w_wide.rename(columns={"lead_hours": "lead_bucket"})

        merged = df.merge(w_wide, on=["zone", "lead_bucket"], how="left", suffixes=("", "_w"))

        blend_val = pd.Series(0.0, index=merged.index)
        weight_sum = pd.Series(0.0, index=merged.index)
        equal_vals = []

        for model, val_col in models_here.items():
            if model not in merged.columns:
                continue  # this model has no skill-table weight for this var (e.g. AIFS wind gusts)
            w = merged[model].fillna(0.0)
            v = merged[val_col]
            has_val = v.notna()
            blend_val += np.where(has_val, w.where(has_val, 0.0) * v.fillna(0), 0.0)
            weight_sum += np.where(has_val, w.where(has_val, 0.0), 0.0)
            equal_vals.append(v)

        # renormalize in case a model was missing for some rows (weights won't sum to 1 there)
        with np.errstate(invalid="ignore", divide="ignore"):
            blend_val = np.where(weight_sum > 0, blend_val / weight_sum.replace(0, np.nan), np.nan)

        equal_mean = pd.concat(equal_vals, axis=1).mean(axis=1)

        result = merged[["point_id", "zone", "site_name", "lat", "lon", "valid_time", "ist_time", "lead_hours", "lead_bucket"]].copy()
        result["variable"] = var
        result[f"{var}_blend"] = blend_val
        result[f"{var}_equal_mean"] = equal_mean
        for model, val_col in models_here.items():
            result[f"{var}_{model}"] = merged[val_col]

        out_frames.append(result)

    # merge the per-variable frames back into one row per (point, valid_time)
    base = out_frames[0].drop(columns=["variable"])
    for f in out_frames[1:]:
        val_cols = [c for c in f.columns if c not in
                    ["point_id", "zone", "site_name", "lat", "lon", "valid_time", "ist_time", "lead_hours", "lead_bucket", "variable"]]
        base = base.merge(f[["point_id", "valid_time"] + val_cols], on=["point_id", "valid_time"], how="left")

    return base


def summary_report(weights: pd.DataFrame, blended: pd.DataFrame):
    print("=" * 70)
    print("BLENDING SUMMARY")
    print("=" * 70)

    print("\n--- Temperature weights at 24h lead, by zone (proof weights actually vary) ---")
    w = weights[(weights["variable"] == "temperature_2m") & (weights["lead_hours"] == 24)]
    pivot = w.pivot_table(index="zone", columns="model", values="weight").round(2)
    print(pivot.to_string())

    print("\n--- Temperature weights at 144h lead, by zone ---")
    w2 = weights[(weights["variable"] == "temperature_2m") & (weights["lead_hours"] == 144)]
    pivot2 = w2.pivot_table(index="zone", columns="model", values="weight").round(2)
    print(pivot2.to_string())

    print(f"\n--- Blended forecast sample ---")
    cols = ["point_id", "zone", "lead_hours", "temperature_2m_blend", "temperature_2m_equal_mean"]
    print(blended[cols].dropna().head(8).to_string(index=False))

    n_total = len(blended)
    n_blended = blended["temperature_2m_blend"].notna().sum()
    print(f"\nBlended {n_blended:,} / {n_total:,} rows for temperature_2m")


def run(alpha: float = 1.5):
    proc_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    skill_path = os.path.join(proc_dir, "skill_table.csv")
    harmonized_path = os.path.join(proc_dir, "harmonized.csv")

    print(f"Loading skill table: {skill_path}")
    skill = pd.read_csv(skill_path)
    print(f"Loading harmonized forecasts: {harmonized_path}")
    harmonized = pd.read_csv(harmonized_path)
    print(f"  {len(harmonized):,} forecast rows")

    weights = compute_weights(skill, alpha=alpha)
    weights_path = os.path.join(proc_dir, "weights_table.csv")
    weights.to_csv(weights_path, index=False)
    print(f"\nWrote weight table: {len(weights):,} rows -> {weights_path}")

    blended = blend_forecast(harmonized, weights)
    blended_path = os.path.join(proc_dir, "blended_forecast.csv")
    blended.to_csv(blended_path, index=False)
    print(f"Wrote blended forecast: {len(blended):,} rows -> {blended_path}")

    summary_report(weights, blended)
    return weights_path, blended_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.5, help="inverse-error exponent (higher = more aggressive weighting toward the best model)")
    args = parser.parse_args()
    run(alpha=args.alpha)