import pandas as pd
import os

# Load the parquet file
df = pd.read_parquet('your_file.parquet')

# Print column names first
print("=== COLUMNS ===")
for col in df.columns.tolist():
    print(f"  {col}")

print("\n=== PER SCENARIO TYPE SUMMARY ===")

# Key metrics to aggregate
metrics = [
    'score',
    'ego_is_comfortable',
    'no_ego_at_fault_collisions', 
    'time_to_collision_within_bound',
    'ego_is_making_progress',
    'drivable_area_compliance',
    'driving_direction_compliance',
    'speed_limit_compliance',
]

# Only keep columns that exist
available = [m for m in metrics if m in df.columns]

# Group by scenario type
summary = df.groupby('scenario_type')[available].mean()

# Add scenario count
summary.insert(0, 'count', df.groupby('scenario_type').size())

# Sort by score ascending (worst first)
if 'score' in summary.columns:
    summary = summary.sort_values('score', ascending=True)

# Round everything
summary = summary.round(3)

# Print with full width
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.3f}'.format)

print(summary.to_string())

print("\n=== FINAL SCORE ===")
print(f"Overall score: {df[df['scenario_type']=='final_score']['score'].values[0]:.4f}")
