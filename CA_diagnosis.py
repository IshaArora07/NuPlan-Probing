"""
diagnose.py — inspect parquet structure before writing analysis scripts
Run: python diagnose.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

RUN_PATH = Path(os.path.expanduser("~/Desktop/Thesis/Training3"))

print("=" * 60)
print("DIAGNOSTIC — parquet structure inspector")
print("=" * 60)

# ── runner_report ────────────────────────────────────────────

print("\n── runner_report.parquet ──")
rr_path = RUN_PATH / "runner_report.parquet"

if rr_path.exists():
    rr = pd.read_parquet(rr_path)
    print(f"Shape     : {rr.shape}")
    print(f"Columns   : {list(rr.columns)}")
    print(f"Dtypes    :\n{rr.dtypes}")
    print(f"\nFirst 3 rows:\n{rr.head(3).to_string()}")
else:
    print("  [WARN] runner_report.parquet not found")

# ── aggregator_metric ────────────────────────────────────────

print("\n── aggregator_metric/ ──")
agg_dir = RUN_PATH / "aggregator_metric"

if agg_dir.exists():
    files = sorted(agg_dir.glob("*.parquet"))
    if not files:
        print("  [INFO] No parquet files found")
    for f in files:
        try:
            df = pd.read_parquet(f)
            print(f"\n  File    : {f.name}")
            print(f"  Shape   : {df.shape}")
            print(f"  Columns : {list(df.columns)}")
            print(f"  Content :\n{df.to_string()}")
        except Exception as e:
            print(f"  [ERROR] Failed to read {f.name}: {e}")
else:
    print("  [WARN] aggregator_metric/ not found")

# ── metrics/ ─────────────────────────────────────────────────

print("\n── metrics/ (first 2 rows of each) ──")
metrics_dir = RUN_PATH / "metrics"

if metrics_dir.exists():
    files = sorted(metrics_dir.glob("*.parquet"))
    if not files:
        print("  [INFO] No metric parquet files found")
    for f in files:
        try:
            df = pd.read_parquet(f)
            print(f"\n  {f.name}")
            print(f"  Shape   : {df.shape}")
            print(f"  Columns : {list(df.columns)}")
            print(f"  Dtypes  : {dict(df.dtypes)}")
            print(f"  Sample  :\n{df.head(2).to_string()}")
        except Exception as e:
            print(f"  [ERROR] Failed to read {f.name}: {e}")
else:
    print("  [WARN] metrics/ not found")
