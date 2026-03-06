import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import torch

labels_file = "scene_labels.jsonl"
anchors_file = "scene_anchors.npy"
cache_dir = Path("/path/to/feature_cache")

anchors = np.load(anchors_file)

class_tokens = defaultdict(list)

with open(labels_file) as f:
    for line in f:
        data = json.loads(line)
        class_tokens[data["emoe_class_id"]].append(data["token"])

trajectories = defaultdict(list)
endpoints = defaultdict(list)

for cls, tokens in class_tokens.items():

    for token in tqdm(tokens[:2000], desc=f"class {cls}"):

        cache_file = cache_dir / f"{token}.pt"

        if not cache_file.exists():
            continue

        sample = torch.load(cache_file)

        traj = sample["targets"]["trajectory"]  # [T,3]

        xy = traj[:, :2].numpy()

        trajectories[cls].append(xy)
        endpoints[cls].append(xy[-1])

plt.figure(figsize=(8,8))

colors = ["blue","red","green","orange","purple","brown","black"]

for cls in endpoints:

    pts = np.array(endpoints[cls])

    plt.scatter(
        pts[:,0],
        pts[:,1],
        s=5,
        alpha=0.5,
        color=colors[cls]
    )

    a = anchors[cls]

    plt.scatter(
        a[:,0],
        a[:,1],
        s=200,
        marker="*",
        color="black"
    )

plt.title("Target Endpoints + Anchors")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
