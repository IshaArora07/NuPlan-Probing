import pandas as pd
import numpy as np

df = pd.read_parquet('your_file.parquet')

# Remove final_score row
df = df[df['scenario_type'] != 'final_score'].copy()

# Convert numeric columns properly
numeric_cols = [
    'score',
    'ego_is_comfortable',
    'no_ego_at_fault_collisions',
    'time_to_collision_within_bound',
    'drivable_area_compliance',
    'ego_is_making_progress',
    'speed_limit_compliance',
]

# Force convert to numeric
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Only keep existing columns
existing = [c for c in numeric_cols if c in df.columns]

# Group and aggregate
summary = df.groupby('scenario_type')[existing].mean().round(3)
summary.insert(0, 'count', df.groupby('scenario_type').size())
summary = summary.sort_values('score')

# Save to CSV
summary.to_csv('t4_breakdown.csv')
print(summary.to_string())
print(f"\nOverall score: {df['score'].mean():.4f}")
