import pandas as pd
import numpy as np

df = pd.read_parquet('your_file.parquet')

# Remove final_score row
df = df[df['scenario_type'] != 'final_score'].copy()

# Only use float64 columns for aggregation
float_cols = [
    'score',
    'ego_is_comfortable',
    'no_ego_at_fault_collisions',
    'time_to_collision_within_bound',
    'drivable_area_compliance',
    'driving_direction_compliance',
    'ego_is_making_progress',
    'ego_progress_along_expert_route',
    'speed_limit_compliance',
]

# Keep only existing float columns
existing = [c for c in float_cols if c in df.columns 
            and df[c].dtype == np.float64]

print("Using columns:", existing)

# Aggregate
summary = df.groupby('scenario_type')[existing].mean().round(3)
summary.insert(0, 'count', df.groupby('scenario_type').size())
summary = summary.sort_values('score')

# Save to CSV
summary.to_csv('t4_breakdown.csv')
print(summary.to_string())
