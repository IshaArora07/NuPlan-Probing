import pandas as pd

df = pd.read_parquet('your_file.parquet')

# Remove final_score row for cleaner analysis
df_scenarios = df[df['scenario_type'] != 'final_score']

# Print exactly these columns in this order
cols = [
    'scenario_type',
    'score',
    'ego_is_comfortable',
    'no_ego_at_fault_collisions', 
    'time_to_collision_within_bound',
    'drivable_area_compliance',
    'ego_is_making_progress',
    'speed_limit_compliance',
]

# Check which exist
existing = [c for c in cols if c in df.columns]
print("Available:", existing)

summary = df_scenarios.groupby('scenario_type')[existing].mean().round(3)
summary['count'] = df_scenarios.groupby('scenario_type').size()
summary = summary.sort_values('score')

# Save to CSV for clean reading
summary.to_csv('t4_scenario_breakdown.csv')
print(summary[['count', 'score', 'ego_is_comfortable', 
               'no_ego_at_fault_collisions',
               'time_to_collision_within_bound']].to_string())
