import pandas as pd
import numpy as np

df = pd.read_parquet('your_file.parquet')

# Look at just ONE scenario type in detail
# Show ALL individual scenario rows for stopping_with_lead
stopping = df[df['scenario_type'] == 'stopping_with_lead']
print(f"stopping_with_lead rows: {len(stopping)}")
print(stopping[['score', 'ego_is_comfortable', 
                'no_ego_at_fault_collisions',
                'time_to_collision_within_bound']].describe())

# Same for high_lateral_acceleration  
lateral = df[df['scenario_type'] == 'high_lateral_acceleration']
print(f"\nhigh_lateral_acceleration rows: {len(lateral)}")
print(lateral[['score', 'ego_is_comfortable',
               'no_ego_at_fault_collisions', 
               'time_to_collision_within_bound']].describe())
