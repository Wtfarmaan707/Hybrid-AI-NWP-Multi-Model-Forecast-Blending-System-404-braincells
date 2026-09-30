"""
scripts/skill.py

STEP 3 of the pipeline: Verification / skill table.

Reads data/processed/history_with_truth.csv (schema in history_schema.py -
works identically whether that file came from simulate_history.py or
fetch_previous_runs.py) and computes, per (model, zone, lead_hours,
variable):

    bias   - mean(forecast - observed)         : systematic over/under
    mae    - mean(|forecast - observed|)        : typical error size
    rmse   - sqrt(mean((forecast-observed)^2))  : penalizes big misses more
    n      - sample count backing the estimate  : so thin buckets can be flagged

This table is the single most important artifact in the whole project -
Step 4's blend weights are computed directly from it. Everything else
(dashboard, weight maps, "we beat the baseline" claims) is a consumer of
this table, so it's worth spending the effort to get it right rather than
rushing to the blend.

Usage:
    python3 scripts/skill.py
"""

import os
import numpy as np
import pandas as pd


def compute_skill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error"] = df["forecast_value"] - df["observed_value"]
    df["abs_error"] = df["error"].abs()
    df["sq_error"] = df["error"] ** 2

    grouped = df.groupby(["model", "zone", "lead_hours", "variable"]).agg(
        bias=("error", "mean"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda s: np.sqrt(s.mean())),
        n=("error", "size"),
    ).reset_index()

    return grouped


def add_best_model_flag(skill: pd.DataFrame) -> pd.DataFrame:
    """For each (zone, lead_hours, variable), flag which model has the
    lowest RMSE. This is the raw material for Step 4's weight maps - a
    literal 'which model wins where' table."""
    skill = skill.copy()
    idx = skill.groupby(["zone", "lead_hours", "variable"])["rmse"].idxmin()
    skill["is_best_rmse"] = False
    skill.loc[idx, "is_best_rmse"] = True
    return skill


def summary_report(skill: pd.DataFrame):
    print("=" * 70)
    print("SKILL TABLE SUMMARY")
    print("=" * 70)

    print(f"\nRows: {len(skill):,} | Models: {sorted(skill['model'].unique())}")
    print(f"Zones: {sorted(skill['zone'].unique())}")
    print(f"Lead hours: {sorted(skill['lead_hours'].unique())}")
    print(f"Variables: {sorted(skill['variable'].unique())}")

    print("\n--- Overall RMSE by model (temperature_2m, averaged across zone/lead) ---")
    temp = skill[skill["variable"] == "temperature_2m"]
    print(temp.groupby("model")["rmse"].mean().round(3).sort_values().to_string())

    print("\n--- Does skill degrade with lead time? (temperature_2m RMSE) ---")
    pivot = temp.groupby(["model", "lead_hours"])["rmse"].mean().unstack("lead_hours").round(2)
    print(pivot.to_string())

    print("\n--- Which model wins each zone at 24h lead? (temperature_2m) ---")
    at24 = temp[temp["lead_hours"] == 24]
    winners = at24.loc[at24.groupby("zone")["rmse"].idxmin(), ["zone", "model", "rmse"]]
    print(winners.sort_values("zone").to_string(index=False))

    print("\n--- Which model wins each zone at 120h lead? (temperature_2m) ---")
    at120 = temp[temp["lead_hours"] == 120]
    winners = at120.loc[at120.groupby("zone")["rmse"].idxmin(), ["zone", "model", "rmse"]]
    print(winners.sort_values("zone").to_string(index=False))

    print("\n--- Precipitation RMSE by zone (all models, lead 48h) ---")
    precip = skill[(skill["variable"] == "precipitation") & (skill["lead_hours"] == 48)]
    print(precip.pivot_table(index="zone", columns="model", values="rmse").round(2).to_string())


def run():
    in_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "history_with_truth.csv")
    print(f"Loading: {in_path}")
    df = pd.read_csv(in_path)
    print(f"  {len(df):,} rows")

    skill = compute_skill(df)
    skill = add_best_model_flag(skill)

    thin = skill[skill["n"] < 5]
    if len(thin):
        print(f"\n[WARN] {len(thin)} (model,zone,lead,var) buckets have fewer than 5 samples - treat with caution")

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "skill_table.csv")
    skill.to_csv(out_path, index=False)
    print(f"\nWrote skill table: {len(skill):,} rows -> {out_path}")

    summary_report(skill)
    return out_path


if __name__ == "__main__":
    run()
