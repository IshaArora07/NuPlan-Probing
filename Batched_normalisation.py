#!/usr/bin/env python3
"""
src/utils/batch_normalize.py

Applies ego-relative normalization to a collated PlutoFeature batch
loaded from cache in global coordinates.
"""

from __future__ import annotations

import math
import torch


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


def normalize_batch(data: dict, hist_steps: int | None = None) -> dict:
    """
    Normalize a batched PlutoFeature from global UTM coords to ego-relative frame.

    Args:
        data: collated feature dict from features["feature"].data
        hist_steps: optional history length including present index.
                    If None, inferred from target or valid_mask.

    Returns:
        normalized data dict
    """
    assert "origin" in data, (
        "data['origin'] missing — cache was not built with first_time=True"
    )
    assert "angle" in data, (
        "data['angle'] missing — cache was not built with first_time=True"
    )

    # ------------------------------------------------------------------
    # Skip if already normalized
    # ------------------------------------------------------------------
    pos_max = data["agent"]["position"].abs().max().item()
    if pos_max < 100.0:
        # already likely ego-relative
        return data

    center_xy = data["origin"].float()
    center_angle = data["angle"].float()

    B = center_xy.shape[0]

    cos_a = torch.cos(center_angle)
    sin_a = torch.sin(center_angle)

    R = torch.stack(
        [
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a, cos_a], dim=-1),
        ],
        dim=1,
    )  # (B, 2, 2)

    def rot_pos(xy: torch.Tensor) -> torch.Tensor:
        shape = xy.shape
        xy_f = xy.float().reshape(B, -1, 2)
        xy_c = xy_f - center_xy.unsqueeze(1)
        xy_r = torch.bmm(xy_c, R.transpose(1, 2))
        return xy_r.reshape(shape)

    def rot_vec(xy: torch.Tensor) -> torch.Tensor:
        shape = xy.shape
        xy_f = xy.float().reshape(B, -1, 2)
        xy_r = torch.bmm(xy_f, R.transpose(1, 2))
        return xy_r.reshape(shape)

    def rot_angle(a: torch.Tensor) -> torch.Tensor:
        extra_dims = a.dim() - 1
        ca = center_angle
        for _ in range(extra_dims):
            ca = ca.unsqueeze(-1)
        return wrap_to_pi(a.float() - ca)

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    agent = data["agent"]

    agent["position"] = rot_pos(agent["position"].clone())
    agent["heading"] = rot_angle(agent["heading"].clone())
    agent["velocity"] = rot_vec(agent["velocity"].clone())

    # ------------------------------------------------------------------
    # Infer hist_steps safely
    # ------------------------------------------------------------------
    if hist_steps is None:
        if "target" in agent:
            T_total = agent["position"].shape[2]
            T_future = agent["target"].shape[2]
            hist_steps = T_total - T_future
        else:
            # fallback for Pluto default 2s @ 0.1s + present
            hist_steps = 21

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
            mp["polygon_orientation"] = rot_angle(
                mp["polygon_orientation"].clone()
            )

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

    fut_pos = pos[:, :, hist_steps:, :]
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
