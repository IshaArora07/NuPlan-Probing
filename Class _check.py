import json
from collections import Counter

labels_path = "scene_labels.jsonl"  # change path if needed

# Scenario types we care about
target_types = [
    "starting_left_turn",
    "starting_right_turn", 
    "high_lateral_acceleration",
    "waiting_for_pedestrian_to_cross",
    "near_multiple_vehicles",
    "changing_lane",
    "stationary_in_traffic",
    "stopping_with_lead",
]

# Count class assignments per scenario type
results = {t: Counter() for t in target_types}
examples = {t: [] for t in target_types}

with open(labels_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        stype = d.get("scenario_type", "")
        
        if stype in target_types:
            cls_id = d.get("emoe_class_id", -1)
            cls_name = d.get("emoe_class_name", "unknown")
            stage = d.get("stage", "unknown")
            results[stype][f"class{cls_id}_{cls_name}"] += 1
            
            # Save first 3 examples per type
            if len(examples[stype]) < 3:
                examples[stype].append({
                    "token": d.get("token", ""),
                    "emoe_class_id": cls_id,
                    "emoe_class_name": cls_name,
                    "stage": stage,
                    "travel_distance_m": d.get("travel_distance_m", 0),
                })

# Print results
print("=" * 70)
print("SCENE CLASSIFICATION BREAKDOWN BY NUPLAN SCENARIO TYPE")
print("=" * 70)

for stype in target_types:
    counter = results[stype]
    total = sum(counter.values())
    
    if total == 0:
        print(f"\n{stype}: NO EXAMPLES FOUND")
        continue
    
    print(f"\n{stype} (total={total}):")
    
    # Sort by count descending
    for cls, count in counter.most_common():
        pct = 100 * count / total
        print(f"  {cls}: {count} ({pct:.1f}%)")
    
    # Show examples
    print(f"  Examples:")
    for ex in examples[stype]:
        print(f"    class={ex['emoe_class_id']} ({ex['emoe_class_name']}) "
              f"stage={ex['stage']} "
              f"dist={ex['travel_distance_m']:.1f}m")

print("\n" + "=" * 70)
print("SUMMARY — Primary class assignment per scenario type:")
print("=" * 70)
for stype in target_types:
    counter = results[stype]
    total = sum(counter.values())
    if total == 0:
        print(f"  {stype}: NO DATA")
        continue
    primary = counter.most_common(1)[0]
    pct = 100 * primary[1] / total
    print(f"  {stype:45s} -> {primary[0]} ({pct:.1f}%)")
