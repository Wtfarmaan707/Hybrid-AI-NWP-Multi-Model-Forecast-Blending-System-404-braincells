"""
scripts/history_schema.py

The contract between "where historical forecast+truth data comes from"
and "how skill gets computed". Both simulate_history.py (fake data, for
building/testing today) and fetch_previous_runs.py (real data, for
later) must produce a CSV with exactly these columns. skill.py never
needs to know or care which one produced its input.

Columns:
    point_id        str    e.g. "himalayan:srinagar"
    zone             str    e.g. "himalayan"
    model            str    e.g. "gfs_global"
    issue_date       str    date the forecast was issued (YYYY-MM-DD)
    lead_hours       int    hours between issue and valid time
    variable         str    "temperature_2m" | "precipitation" | "wind_speed_10m"
    forecast_value   float  what the model predicted
    observed_value   float  what actually happened (ground truth)

One row = one (point, model, issue_date, lead_hours, variable) forecast
verified against one observation.
"""

REQUIRED_COLUMNS = [
    "point_id", "zone", "model", "issue_date", "lead_hours",
    "variable", "forecast_value", "observed_value",
]

VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m"]

# Standard lead-time buckets used everywhere downstream (skill table,
# blending, dashboard). Keeping this fixed means Step 4's weight lookup
# is a simple bucket match, not a nearest-neighbour search.
LEAD_BUCKETS = [0, 6, 12, 24, 48, 72, 96, 120, 144]


def validate(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"history dataframe missing required columns: {missing}")
    bad_vars = set(df["variable"].unique()) - set(VARIABLES)
    if bad_vars:
        raise ValueError(f"unexpected variable names: {bad_vars} (expected one of {VARIABLES})")
    return True
