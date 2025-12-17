import json
from collections import Counter

path = "scene_labels.jsonl"

c_stage = Counter()
c_round_stage = Counter()
c_pudo_trav = Counter()
c_pudo_best = Counter()

with open(path, "r") as f:
    for line in f:
        r = json.loads(line)
        c_stage[r["stage"]] += 1

        dbg = r.get("debug", {})
        pudo_trav["traversed"] if False else None
        c_pudo_trav[bool(dbg.get("pudo_traversed", False))] += 1
        c_pudo_best[str(dbg.get("pudo_best_type", "NONE"))] += 1

        if r["emoe_class_id"] == 4:
            c_round_stage[r["stage"]] += 1

print("Top stages:", c_stage.most_common(20))
print("Roundabout stages:", c_round_stage.most_common(20))
print("PUDO traversed counts:", c_pudo_trav)
print("PUDO best_type counts:", c_pudo_best.most_common(10))
