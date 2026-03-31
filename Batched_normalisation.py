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


def normalize_batch(data: dict, hist_steps: int = None) -> dict:
    """
    Normalize a batched PlutoFeature from global UTM coords to ego-relative frame.

    Args:
        data: collated feature dict from features["feature"].data
        hist_steps: optional history length including present index.
                    If None, inferred from target shape.

    Returns:
        normalized data dict
    """
    assert "origin" in data, "data['origin'] missing"
    assert "angle" in data, "data['angle'] missing"

    center_xy = data["origin"].float()   # (B, 2)
    center_angle = data["angle"].float() # (B,)

    # ------------------------------------------------------------------
    # Safer skip: check actual agent positions, not stored origin
    # ------------------------------------------------------------------
    agent_pos_mag = data["agent"]["position"].abs().max().item()
    if agent_pos_mag < 100.0:
        return data

    B = center_xy.shape[0]
    device = center_xy.device

    cos_a = torch.cos(center_angle)
    sin_a = torch.sin(center_angle)

    # ------------------------------------------------------------------
    # Rotation matrix
    # row-vector convention: xy_centered @ R
    # ------------------------------------------------------------------
    R = torch.zeros(B, 2, 2, device=device)
    R[:, 0, 0] = cos_a
    R[:, 0, 1] = -sin_a
    R[:, 1, 0] = sin_a
    R[:, 1, 1] = cos_a

    def rot_pos(xy: torch.Tensor) -> torch.Tensor:
        shape = xy.shape
        xy_f = xy.float().reshape(B, -1, 2)
        xy_c = xy_f - center_xy.unsqueeze(1)
        xy_r = torch.bmm(xy_c, R)
        return xy_r.reshape(shape)

    def rot_vec(xy: torch.Tensor) -> torch.Tensor:
        shape = xy.shape
        xy_f = xy.float().reshape(B, -1, 2)
        xy_r = torch.bmm(xy_f, R)
        return xy_r.reshape(shape)

    def rot_angle(a: torch.Tensor) -> torch.Tensor:
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

    # ------------------------------------------------------------------
    # Infer hist_steps
    # ------------------------------------------------------------------
    if hist_steps is None:
        if "target" in agent:
            T_total = agent["position"].shape[2]
            T_future = agent["target"].shape[2]
            hist_steps = T_total - T_future
        else:
            hist_steps = 21  # PLUTO default

    _recompute_target(data, hist_steps)

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
            pc = mp["polygon_center"].float().clone()
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
    """
    pos = data["agent"]["position"].float()
    hdg = data["agent"]["heading"].float()
    vmask = data["agent"]["valid_mask"]

    T_total = pos.shape[2]
    hist_steps = min(hist_steps, T_total - 1)

    origin_pos = pos[:, :, hist_steps - 1, :]
    origin_hdg = hdg[:, :, hist_steps - 1]

    fut_pos = pos[:, :, hist_steps:]
    fut_hdg = hdg[:, :, hist_steps:]

    tgt_pos = fut_pos - origin_pos.unsqueeze(2)
    tgt_hdg = wrap_to_pi(fut_hdg - origin_hdg.unsqueeze(2))

    target = torch.cat(
        [tgt_pos, tgt_hdg.unsqueeze(-1)],
        dim=-1,
    )

    fut_mask = vmask[:, :, hist_steps:]
    target = target * fut_mask.unsqueeze(-1).float()

    data["agent"]["target"] = target
