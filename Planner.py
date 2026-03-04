routing_idx = out["routing_idx"][0].item()
router_logits = out["router_logits"][0].cpu()
router_probs = torch.softmax(router_logits, dim=-1)

scene_type_names = {
    0: "left_turn_at_intersection",
    1: "straight_at_intersection",
    2: "right_turn_at_intersection",
    3: "straight_non_intersection",
    4: "roundabout_or_uturn",
    5: "others",
}

print(
    f"[Router] step={current_input.iteration.index:04d} | "
    f"scene={scene_type_names[routing_idx]} (expert {routing_idx}) | "
    f"probs={[f'{p:.2f}' for p in router_probs.tolist()]} | "
    f"best_mode={out['routing_idx'][0].item()}"
)
