def lane_continuity_gate(
    cache: Dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    hs: np.ndarray,
    *,
    sample_step: int,
    match_radius_m: float,
    heading_gate_deg: float,
    min_hits: int,
) -> Dict[str, Any]:
    """
    Map-match ego samples to LANE geometries and estimate whether ego stays within
    a consistent lane-group/roadblock corridor.

    Returns dict with:
      - ok: bool
      - dominant_ratio: float
      - unique_groups: int
      - hits: int
      - group_col: str
    """
    lane_geoms = cache.get("lane_geoms", [])
    lane_sindex = cache.get("lane_sindex", None)
    group_vals = cache.get("lane_group_vals", None)
    group_col = cache.get("lane_group_col", "")

    if not lane_geoms or group_vals is None or len(group_vals) == 0 or not group_col:
        return {"ok": False, "dominant_ratio": 0.0, "unique_groups": 0, "hits": 0, "group_col": group_col}

    r = float(match_radius_m)
    heading_gate = float(heading_gate_deg)

    groups = []
    hits = 0

    for i in range(0, len(xs), max(1, sample_step)):
        p = Point(float(xs[i]), float(ys[i]))
        ego_h = float(hs[i]) if i < len(hs) else float(hs[-1])

        if lane_sindex is not None:
            bbox = (p.x - r, p.y - r, p.x + r, p.y + r)
            cand_idx = list(lane_sindex.intersection(bbox))
        else:
            cand_idx = list(range(len(lane_geoms)))

        best_j = None
        best_d = r

        for j in cand_idx:
            if j >= len(lane_geoms):
                continue
            g = lane_geoms[j]
            if g is None:
                continue
            try:
                d = float(g.distance(p))
            except Exception:
                continue
            if d <= best_d:
                if heading_gate > 0:
                    gh = approx_geom_heading(g, p)
                    if gh is not None:
                        dh = abs(wrap_to_pi(gh - ego_h))
                        if math.degrees(dh) > heading_gate:
                            continue
                best_d = d
                best_j = j

        if best_j is None:
            continue

        hits += 1
        if best_j < len(group_vals):
            groups.append(group_vals[best_j])

    if hits < int(min_hits) or len(groups) == 0:
        return {"ok": False, "dominant_ratio": 0.0, "unique_groups": 0, "hits": int(hits), "group_col": group_col}

    counts = Counter(groups)
    dominant = max(counts.values())
    dominant_ratio = dominant / max(1, len(groups))
    unique_groups = len(counts)

    ok = (dominant_ratio >= CURVED_STRAIGHT_MIN_DOMINANT_LANE_RATIO) and (unique_groups <= CURVED_STRAIGHT_MAX_UNIQUE_LANE_GROUPS)

    return {
        "ok": bool(ok),
        "dominant_ratio": float(dominant_ratio),
        "unique_groups": int(unique_groups),
        "hits": int(hits),
        "group_col": str(group_col),
    }













    # Curvature distribution: detect "smooth curvature" vs a localized turn
    dh_steps = np.diff(headings)
    dh_steps = np.vectorize(wrap_to_pi)(dh_steps)
    abs_step_deg = np.degrees(np.abs(dh_steps)) if len(dh_steps) > 0 else np.array([0.0])
    max_step_abs_deg = float(np.max(abs_step_deg)) if abs_step_deg.size > 0 else 0.0
    p95_step_abs_deg = float(np.percentile(abs_step_deg, 95)) if abs_step_deg.size > 1 else max_step_abs_deg







    lane_cont = lane_continuity_gate(
        cache, xs, ys, headings,
        sample_step=lane_sample_step,
        match_radius_m=lane_match_radius_m,
        heading_gate_deg=lane_heading_gate_deg,
        min_hits=lane_min_hits,
    )
    debug["lane_continuity"] = lane_cont
    lane_cont_ok = bool(lane_cont.get("ok", False))











        if NET_TURN_MIN_AT_INTERSECTION <= abs_dh <= NET_TURN_MAX_AT_INTERSECTION:
            # (2) Connector veto: if connector confidently says STRAIGHT, do NOT allow left/right.
            bt = str(conn_mm.get("best_type", "NONE"))
            br = float(conn_mm.get("best_ratio", 0.0))
            connector_confident = traversed_connector and (br >= float(connector_verify_min_ratio))
            connector_says_turn = connector_confident and (bt in ("LEFT", "RIGHT"))
            connector_says_straight = connector_confident and (bt == "STRAIGHT")

            if connector_says_straight:
                debug["connector_veto_to_straight"] = {"best_type": bt, "best_ratio": br}
                return 1, "stage3_intersection_connector_veto_straight", debug

            # (1)+(3) Curved-lane-following gate:
            # If curvature is smooth and ego stays in same lane-corridor, treat as straight-through.
            if lane_following_ok and lane_cont_ok and smooth_curvature and (not connector_says_turn):
                debug["curved_lane_following_promoted_to_straight"] = {
                    "lane_following_ok": bool(lane_following_ok),
                    "lane_cont_ok": bool(lane_cont_ok),
                    "smooth_curvature": bool(smooth_curvature),
                    "max_step_abs_heading_deg": max_step_abs_deg,
                    "connector_best_type": bt,
                    "connector_best_ratio": br,
                }
                return 1, "stage3_intersection_curved_lane_following", debug

            # Otherwise: it is a real turn candidate by net heading
            geom_cls = 0 if delta_heading > 0.0 else 2  # left if positive, else right

            if traversed_connector and connector_confident:
                debug["connector_verify_note"] = f"best={bt}, ratio={br:.2f}"
                if (geom_cls == 0 and bt == "RIGHT") or (geom_cls == 2 and bt == "LEFT"):
                    debug["connector_verify_conflict"] = True

            return geom_cls, "stage3_intersection_net_heading", debug









# Stronger "STRAIGHT veto": only when we are confident it's a curved-straight case
STRAIGHT_VETO_MIN_RATIO = 0.80          # raise from ~0.55
STRAIGHT_VETO_MAX_ABS_DH_DEG = 55.0     # don't veto real turns (>~60°)

if connector_confident and bt == "STRAIGHT" and br >= STRAIGHT_VETO_MIN_RATIO:
    if (math.degrees(abs_dh) <= STRAIGHT_VETO_MAX_ABS_DH_DEG) and lane_following_ok and lane_cont_ok and smooth_curvature:
        debug["connector_veto_to_straight"] = {
            "best_type": bt, "best_ratio": br,
            "abs_dh_deg": math.degrees(abs_dh),
            "lane_following_ok": lane_following_ok,
            "lane_cont_ok": lane_cont_ok,
            "smooth_curvature": smooth_curvature,
        }
        return 1, "stage3_intersection_connector_veto_straight", debug
    else:
        debug["connector_veto_blocked"] = {
            "best_type": bt, "best_ratio": br,
            "abs_dh_deg": math.degrees(abs_dh),
            "lane_following_ok": lane_following_ok,
            "lane_cont_ok": lane_cont_ok,
            "smooth_curvature": smooth_curvature,
        }



