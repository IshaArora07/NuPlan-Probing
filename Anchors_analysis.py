import numpy as np
import torch

original = np.load("scene_anchors.npy")
ckpt = torch.load("t4_checkpoint.ckpt", map_location="cpu")
learned = ckpt["state_dict"][
    "model.mode_query_generator.anchors_xy"
].numpy()

for c in range(6):
    print(f"\n=== Class {c} ===")
    print(f"Original: x=[{original[c,:,0].min():.2f}, {original[c,:,0].max():.2f}] "
          f"y=[{original[c,:,1].min():.3f}, {original[c,:,1].max():.3f}] "
          f"x_std={original[c,:,0].std():.3f} y_std={original[c,:,1].std():.3f}")
    print(f"Learned:  x=[{learned[c,:,0].min():.2f}, {learned[c,:,0].max():.2f}] "
          f"y=[{learned[c,:,1].min():.3f}, {learned[c,:,1].max():.3f}] "
          f"x_std={learned[c,:,0].std():.3f} y_std={learned[c,:,1].std():.3f}")
    
    # Specifically check for stopping anchors
    orig_stopping = (original[c,:,0] < 10).sum()
    learned_stopping = (learned[c,:,0] < 10).sum()
    print(f"Stopping anchors (x<10m): original={orig_stopping}, learned={learned_stopping}")
    
    # Check lateral anchors
    orig_lateral = (np.abs(original[c,:,1]) > 1.0).sum()
    learned_lateral = (np.abs(learned[c,:,1]) > 1.0).sum()
    print(f"Lateral anchors (|y|>1m): original={orig_lateral}, learned={learned_lateral}")
