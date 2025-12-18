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
