import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import json

def generate_t6_anchors(
    labels_path: str,
    output_path: str,
    Ka: int = 24,
    num_classes: int = 6,
    x_cap: float = 40.0,        # cap anchors at 8-second horizon
    min_travel: float = 1.0,    # include slow/stopping scenarios
    kmeans_seed: int = 42,
):
    """
    T6 anchor generation with:
    1. x capped at 40m (8-second horizon)
    2. Explicit stopping anchors for classes 3, 1
    3. Explicit lateral anchors for class 3
    4. KMeans on remaining slots
    """
    
    # ---- Load endpoints from jsonl ----
    endpoints_by_class = defaultdict(list)
    with open(labels_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            cls = rec.get("emoe_class_id", -1)
            if not (0 <= cls < num_classes):
                continue
            dist = rec.get("travel_distance_m", 0.0)
            if dist < min_travel:  # include stopping scenarios
                continue
            ep = rec.get("endpoint_xy", None)
            if ep is None or len(ep) != 2:
                continue
            x, y = float(ep[0]), float(ep[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            # Cap x at horizon
            x = np.clip(x, -x_cap, x_cap)
            endpoints_by_class[cls].append([x, y])
    
    scene_anchors = np.zeros((num_classes, Ka, 2), dtype=np.float32)
    
    for c in range(num_classes):
        pts = np.array(endpoints_by_class[c], dtype=np.float32)
        
        # ---- Define mandatory anchors per class ----
        mandatory = []
        
        if c == 3:  # straight non-intersection — most critical
            # Stopping anchors (for stopping_with_lead, 
            # stationary_in_traffic, low_magnitude_speed)
            mandatory += [
                [2.0,  0.0],   # nearly stopped
                [4.0,  0.0],   # very slow
                [6.0,  0.0],   # slow
                [8.0,  0.0],   # slow-medium
                [10.0, 0.0],   # medium-slow
                [12.0, 0.0],   # medium
            ]
            # Lateral anchors (for changing_lane, 
            # near_multiple_vehicles)
            mandatory += [
                [20.0, -3.7],  # lane change right
                [20.0, -2.5],  # slight right
                [20.0, -1.5],  # nudge right
                [20.0,  1.5],  # nudge left
                [20.0,  2.5],  # slight left
                [20.0,  3.7],  # lane change left
                [30.0, -3.7],  # lane change right far
                [30.0,  3.7],  # lane change left far
            ]
            
        elif c == 1:  # straight at intersection
            # Stopping at traffic light/sign
            mandatory += [
                [2.0,  0.0],
                [5.0,  0.0],
                [8.0,  0.0],
                [12.0, 0.0],
            ]
            # Slight lateral for intersection navigation
            mandatory += [
                [20.0, -2.0],
                [20.0,  2.0],
            ]
            
        elif c == 0:  # left turn
            # Stopping before turn
            mandatory += [
                [2.0, 0.0],
                [5.0, 0.0],
            ]
            
        elif c == 2:  # right turn
            # Stopping before turn
            mandatory += [
                [2.0,  0.0],
                [5.0,  0.0],
            ]
        
        mandatory = np.array(mandatory, dtype=np.float32) \
            if mandatory else np.zeros((0, 2), dtype=np.float32)
        n_mandatory = len(mandatory)
        n_kmeans = Ka - n_mandatory
        
        # ---- KMeans on remaining slots ----
        if n_kmeans > 0 and len(pts) > 0:
            n_clusters = min(n_kmeans, len(pts))
            km = KMeans(
                n_clusters=n_clusters,
                random_state=kmeans_seed,
                n_init=10,
                max_iter=500,
            )
            km.fit(pts)
            centers = km.cluster_centers_.astype(np.float32)
            
            # Pad if needed with noise
            if n_clusters < n_kmeans:
                rng = np.random.RandomState(kmeans_seed)
                extra_idx = rng.choice(
                    n_clusters, 
                    size=n_kmeans - n_clusters, 
                    replace=True
                )
                noise = rng.randn(
                    n_kmeans - n_clusters, 2
                ).astype(np.float32) * 1.0
                extra = centers[extra_idx] + noise
                centers = np.concatenate([centers, extra], axis=0)
        else:
            rng = np.random.RandomState(kmeans_seed)
            centers = rng.randn(n_kmeans, 2).astype(np.float32)
        
        # ---- Combine mandatory + KMeans ----
        if n_mandatory > 0:
            all_anchors = np.concatenate(
                [mandatory, centers[:n_kmeans]], axis=0
            )
        else:
            all_anchors = centers[:Ka]
        
        scene_anchors[c] = all_anchors[:Ka]
        
        # ---- Print summary ----
        print(f"\nClass {c}:")
        print(f"  Mandatory anchors: {n_mandatory}")
        print(f"  KMeans anchors: {n_kmeans} "
              f"(from {len(pts)} pts)")
        print(f"  Final x range: "
              f"[{scene_anchors[c,:,0].min():.2f}, "
              f"{scene_anchors[c,:,0].max():.2f}]")
        print(f"  Final y range: "
              f"[{scene_anchors[c,:,1].min():.2f}, "
              f"{scene_anchors[c,:,1].max():.2f}]")
        stopping = (scene_anchors[c,:,0] < 10).sum()
        lateral = (np.abs(scene_anchors[c,:,1]) > 1.0).sum()
        print(f"  Stopping anchors (x<10m): {stopping}")
        print(f"  Lateral anchors (|y|>1m): {lateral}")
    
    np.save(output_path, scene_anchors)
    print(f"\nSaved anchors to {output_path}")
    print(f"Shape: {scene_anchors.shape}")
    return scene_anchors


if __name__ == "__main__":
    anchors = generate_t6_anchors(
        labels_path="scene_labels.jsonl",
        output_path="scene_anchors_t6.npy",
        Ka=24,
        num_classes=6,
        x_cap=40.0,
        min_travel=1.0,
    )
