import pandas as pd
import numpy as np

df = pd.read_parquet('your_file.parquet')

for stype in ['high_lateral_acceleration', 
              'starting_left_turn',
              'starting_right_turn',
              'near_multiple_vehicles',
              'stopping_with_lead']:
    
    subset = df[df['scenario_type'] == stype]
    
    # Count scenarios with score = 0
    zero_scores = (subset['score'] == 0).sum()
    low_scores = (subset['score'] < 0.5).sum()
    
    # Check which metric is failing
    collision_fails = (subset['no_ego_at_fault_collisions'] == 0).sum()
    ttc_fails = (subset['time_to_collision_within_bound'] == 0).sum()
    comfort_fails = (subset['ego_is_comfortable'] == 0).sum()
    drivable_fails = (subset['drivable_area_compliance'] == 0).sum()
    
    print(f"\n{stype} (n={len(subset)}):")
    print(f"  score=0: {zero_scores} | score<0.5: {low_scores}")
    print(f"  collision_fails: {collision_fails}")
    print(f"  ttc_fails: {ttc_fails}")
    print(f"  comfort_fails: {comfort_fails}")
    print(f"  drivable_fails: {drivable_fails}")
