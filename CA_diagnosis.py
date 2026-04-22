“””
diagnose.py — inspect parquet structure before writing analysis scripts
Run: python diagnose.py
“””
import os, sys
import pandas as pd
from pathlib import Path

RUN_PATH = Path(os.path.expanduser(”~/Desktop/Thesis/Training3”))

print(”=” * 60)
print(“DIAGNOSTIC — parquet structure inspector”)
print(”=” * 60)

# ── runner_report ────────────────────────────────────────────

print(”\n── runner_report.parquet ──”)
rr = pd.read_parquet(RUN_PATH / “runner_report.parquet”)
print(f”Shape     : {rr.shape}”)
print(f”Columns   : {list(rr.columns)}”)
print(f”Dtypes    :\n{rr.dtypes}”)
print(f”\nFirst 3 rows:\n{rr.head(3).to_string()}”)

# ── aggregator_metric ────────────────────────────────────────

print(”\n── aggregator_metric/ ──”)
agg_dir = RUN_PATH / “aggregator_metric”
for f in sorted(agg_dir.glob(”*.parquet”)):
df = pd.read_parquet(f)
print(f”\n  File    : {f.name}”)
print(f”  Shape   : {df.shape}”)
print(f”  Columns : {list(df.columns)}”)
print(f”  Content :\n{df.to_string()}”)

# ── metrics/ ─────────────────────────────────────────────────

print(”\n── metrics/ (first 2 rows of each) ──”)
metrics_dir = RUN_PATH / “metrics”
for f in sorted(metrics_dir.glob(”*.parquet”)):
df = pd.read_parquet(f)
print(f”\n  {f.name}”)
print(f”  Shape   : {df.shape}”)
print(f”  Columns : {list(df.columns)}”)
print(f”  Dtypes  : {dict(df.dtypes)}”)
print(f”  Sample  :\n{df.head(2).to_string()}”)
