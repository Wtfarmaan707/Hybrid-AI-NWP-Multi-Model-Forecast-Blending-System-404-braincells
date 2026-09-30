"""
scripts/run_all.py

Runs the entire pipeline end-to-end, in order, with one command. This is
the "automated workflow" deliverable from the original brief - instead
of running 6 scripts by hand, run this once.

Usage:
    python scripts/run_all.py                          # uses latest raw ingestion file
    python scripts/run_all.py --raw data/raw/forecast_X.csv   # specify which raw file to harmonize
    python scripts/run_all.py --skip-ingest             # skip Step 1 (use existing raw data)
    python scripts/run_all.py --real-history             # use fetch_previous_runs.py instead of simulate_history.py

What it runs, in order:
    1. ingest.py            (skippable with --skip-ingest)
    2. harmonize.py          (on the newest file in data/raw/, or --raw)
    3. simulate_history.py OR fetch_previous_runs.py (--real-history)
    4. skill.py
    5. blend.py
    6. extremes.py
    7. confidence.py

After this finishes, dashboard.html (served via `python -m http.server`)
will show fully up-to-date results.
"""

import argparse
import glob
import os
import subprocess
import sys
import time

SCRIPTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(SCRIPTS_DIR, "..")


def run_step(label, cmd):
    print(f"\n{'='*70}\nSTEP: {label}\n{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {label} exited with code {result.returncode} after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"[OK] {label} finished in {elapsed:.1f}s")


def latest_raw_file():
    files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "raw", "forecast_*.csv")))
    if not files:
        return None
    return files[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", help="specific raw CSV to harmonize (default: newest in data/raw/)")
    parser.add_argument("--skip-ingest", action="store_true", help="skip Step 1, reuse existing raw data")
    parser.add_argument("--real-history", action="store_true",
                         help="use fetch_previous_runs.py (real data, needs network) instead of simulate_history.py")
    parser.add_argument("--days", type=int, default=14, help="days of simulated history (ignored with --real-history)")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_ingest:
        run_step("1. Ingestion (live forecast pull)", [py, "scripts/ingest.py"])

    raw_file = args.raw or latest_raw_file()
    if not raw_file:
        print("[FAILED] No raw ingestion file found in data/raw/. Run without --skip-ingest first.")
        sys.exit(1)
    run_step(f"2. Harmonization ({os.path.basename(raw_file)})", [py, "scripts/harmonize.py", raw_file])

    if args.real_history:
        run_step("3. Historical truth (real, via Open-Meteo Previous Runs)", [py, "scripts/fetch_previous_runs.py"])
    else:
        run_step("3. Historical truth (simulated)", [py, "scripts/simulate_history.py", "--days", str(args.days)])

    run_step("3b. Skill table", [py, "scripts/skill.py"])
    run_step("4. Blending engine", [py, "scripts/blend.py"])
    run_step("5a. Extreme events", [py, "scripts/extremes.py"])
    run_step("5b. Confidence index", [py, "scripts/confidence.py"])

    print(f"\n{'='*70}\nPIPELINE COMPLETE\n{'='*70}")
    print("All 6 steps ran successfully. To view the dashboard:")
    print("  python -m http.server 8000")
    print("  then open http://localhost:8000/dashboard.html")


if __name__ == "__main__":
    main()
