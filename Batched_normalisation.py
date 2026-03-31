#!/usr/bin/env python3
"""
src/utils/batch_normalize.py

Applies ego-relative normalization to a collated PlutoFeature batch
loaded from cache in global coordinates.

Matches PlutoFeature.normalize() exactly:
rotated = (pos - center_xy) @ rotate_mat
where rotate_mat = [[cos, -sin], [sin, cos]]
"""

from __future__ import annotations

import math
import torch


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


def normalize_batch(data: dict, hist_steps: int = None, debug: bool = False) -> dict:
    """
    Normalize a batched PlutoFeature from global UTM coords to ego-relative frame.

    Args:
        data: collated feature dict from features["feature"].data
        hist_steps: optional history length including present index.
                    If None, inferred from target shape.
        debug: if True, print diagnostics

    Returns:
        normalized data dict
    """
    assert "origin" in data, "data['origin'] missing"
    assert "angle" in data, "data['angle'] missing"

    center_xy = data["origin"].float()   # (B, 2)
    center_angle = data["angle"].float()  # (B,)

    # ------------------------------------------------------------------
    # Skip if already normalized
    # ------------------------------------------------------------------
    agent_pos_mag = data["agent"]["position"].abs().max().item()
    if agent_pos_mag < 100.0:
        if debug:
            print(
                f"[NORM] Skipping — agent_pos_mag={agent_pos_mag:.2f} < 100, already normalized"
            )
        return data

    B = center_xy.shape[0]
    device = center_xy.device

    cos_a = torch.cos(center_angle)
    sin_a = torch.sin(center_angle)

    # ------------------------------------------------------------------
    # Rotation matrix — row-vector convention: xy_centered @ R
    # R[b] = [[cos, -sin],
    #         [sin,  cos]]
    # ------------------------------------------------------------------
    R = torch.zeros(B, 2, 2, device=device)
    R[:, 0, 0] = cos_a
    R[:, 0, 1] = -sin_a
    R[:, 1, 0] = sin_a
    R[:, 1, 1] = cos_a

    # ------------------------------------------------------------------
    # DEBUG: verify rotation on a single point before any reshape
    # ------------------------------------------------------------------
    if debug:
        print(f"\n{'=' * 50}")
        print(f"[NORM DEBUG] B={B}, device={device}")
        print(f"[NORM DEBUG] center_xy[0]    = {center_xy[0].tolist()}")
        print(f"[NORM DEBUG] center_angle[0] = {center_angle[0].item():.4f} rad")
        print(
            f"[NORM DEBUG] cos_a[0]={cos_a[0].item():.4f}, sin_a[0]={sin_a[0].item():.4f}"
        )
        print(f"[NORM DEBUG] R[0] =\n{R[0]}")

        present_idx = 20
        p0 = data["agent"]["position"][0, 0, present_idx, :].float()
        c0 = center_xy[0]
        diff = p0 - c0
        rot = diff @ R[0]
        print(f"[NORM DEBUG] p0 (ego present)   = {p0.tolist()}")
        print(f"[NORM DEBUG] c0 (center_xy)      = {c0.tolist()}")
        print(f"[NORM DEBUG] diff = p0 - c0       = {diff.tolist()}")
        print(f"[NORM DEBUG] rotated = diff @ R[0]= {rot.tolist()}")
        print(
            f"[NORM DEBUG] agent pos max BEFORE = {data['agent']['position'].abs().max().item():.4f}"
        )
        print(f"{'=' * 50}\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def rot_pos(xy: torch.Tensor) -> torch.Tensor:
        """Subtract UTM center then rotate. xy: (B, ..., 2)"""
        shape = xy.shape
        xy_f = xy.float().reshape(B, -1, 2)
        xy_c = xy_f - center_xy.unsqueeze(1)
        xy_r = torch.bmm(xy_c, R)
        return xy_r.reshape(shape)

    def rot_vec(xy: torch.Tensor) -> torch.Tensor:
        """Rotate only, no translation. xy: (B, ..., 2)"""
        shape = xy.shape
        xy_f = xy.float().reshape(B, -1, 2)
        xy_r = torch.bmm(xy_f, R)
        return xy_r.reshape(shape)

    def rot_angle(a: torch.Tensor) -> torch.Tensor:
        """Subtract center_angle and wrap to [-pi, pi]. a: (B, ...)"""
        ca = center_angle
        for _ in range(a.dim() - 1):
            ca = ca.unsqueeze(-1)
        return wrap_to_pi(a.float() - ca)

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    agent = data["agent"]

    agent["position"] = rot_pos(agent["position"].clone())
    agent["heading"] = rot_angle(agent["heading"].clone())
    agent["velocity"] = rot_vec(agent["velocity"].clone())

    if "acceleration" in agent:
        agent["acceleration"] = rot_vec(agent["acceleration"].clone())

    if debug:
        print(
            f"[NORM DEBUG] agent pos max AFTER rot_pos = {agent['position'].abs().max().item():.4f}"
        )
        print(
            f"[NORM DEBUG] agent pos[0,0,20,:] AFTER   = {agent['position'][0, 0, 20, :].tolist()}"
        )

    # ------------------------------------------------------------------
    # Infer hist_steps
    # ------------------------------------------------------------------
    if hist_steps is None:
        if "target" in agent:
            T_total = agent["position"].shape[2]
            T_future = agent["target"].shape[2]
            hist_steps = T_total - T_future
        else:
            hist_steps = 21  # PLUTO default: 2s @ 0.1s + present

    if debug:
        print(f"[NORM DEBUG] hist_steps = {hist_steps}")

    _recompute_target(data, hist_steps)

    if debug:
        print(
            f"[NORM DEBUG] target max AFTER recompute = {data['agent']['target'].abs().max().item():.4f}"
        )

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------
    if "map" in data:
        mp = data["map"]

        if "point_position" in mp:
            mp["point_position"] = rot_pos(mp["point_position"].clone())
        if "point_vector" in mp:
            mp["point_vector"] = rot_vec(mp["point_vector"].clone())
        if "point_orientation" in mp:
            mp["point_orientation"] = rot_angle(mp["point_orientation"].clone())

        if "polygon_center" in mp:
            pc = mp["polygon_center"].float().clone()   # (B, M, 3)
            pc[..., :2] = rot_pos(pc[..., :2])
            pc[..., 2] = rot_angle(pc[..., 2])
            mp["polygon_center"] = pc

        if "polygon_position" in mp:
            mp["polygon_position"] = rot_pos(mp["polygon_position"].clone())
        if "polygon_orientation" in mp:
            mp["polygon_orientation"] = rot_angle(mp["polygon_orientation"].clone())

    # ------------------------------------------------------------------
    # Static objects
    # ------------------------------------------------------------------
    if "static_objects" in data:
        so = data["static_objects"]
        if "position" in so:
            so["position"] = rot_pos(so["position"].clone())
        if "heading" in so:
            so["heading"] = rot_angle(so["heading"].clone())

    # ------------------------------------------------------------------
    # Current state
    # ------------------------------------------------------------------
    if "current_state" in data:
        cs = data["current_state"].float().clone()
        cs[:, :3] = 0.0
        data["current_state"] = cs

    return data


def _recompute_target(data: dict, hist_steps: int) -> None:
    """
    Recompute data["agent"]["target"] from normalized position + heading.

    Matches PlutoFeature.normalize() exactly:
        target_pos = position[:, hist_steps:] - position[:, hist_steps-1, None]
        target_hdg = heading[:, hist_steps:]  - heading[:, hist_steps-1, None]
        target     = concat([target_pos, target_hdg], dim=-1)
        target[~valid_mask] = 0

    Output: (B, A, T_future, 3)
    """
    pos = data["agent"]["position"].float()   # (B, A, T_total, 2)
    hdg = data["agent"]["heading"].float()    # (B, A, T_total)
    vmask = data["agent"]["valid_mask"]       # (B, A, T_total)

    T_total = pos.shape[2]
    hist_steps = min(hist_steps, T_total - 1)

    origin_pos = pos[:, :, hist_steps - 1, :]   # (B, A, 2)
    origin_hdg = hdg[:, :, hist_steps - 1]      # (B, A)

    fut_pos = pos[:, :, hist_steps:]            # (B, A, T_future, 2)
    fut_hdg = hdg[:, :, hist_steps:]            # (B, A, T_future)

    tgt_pos = fut_pos - origin_pos.unsqueeze(2)               # (B, A, T_future, 2)
    tgt_hdg = wrap_to_pi(fut_hdg - origin_hdg.unsqueeze(2))   # (B, A, T_future)

    target = torch.cat(
        [tgt_pos, tgt_hdg.unsqueeze(-1)],
        dim=-1,
    )  # (B, A, T_future, 3)

    fut_mask = vmask[:, :, hist_steps:]
    target = target * fut_mask.unsqueeze(-1).float()
    data["agent"]["target"] = target
