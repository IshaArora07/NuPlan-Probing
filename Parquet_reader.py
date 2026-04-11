import pandas as pd

df = pd.read_parquet('your_file.parquet')

# Group by scenario type and get mean of key metrics
summary = df.groupby('scenario_type').agg({
    'score': 'mean',
    'ego_is_comfortable': 'mean',
    'no_ego_at_fault_collisions': 'mean',
    'time_to_collision_within_bound': 'mean',
    'ego_is_making_progress': 'mean',
    'num_scenarios': 'sum'
}).round(3)

print(summary.to_string())
