#!/usr/bin/env python3
"""
src/utils/batch_normalize.py

Applies ego-relative normalization to a collated PlutoFeature batch
loaded from cache in global coordinates.

Usage in pluto_trainer.py _step():
from src.utils.batch_normalize import normalize_batch

def _step(self, batch, prefix):
    features, targets, scenarios = batch
    data = features["feature"].data
    data = normalize_batch(data)          # <-- add this line
    res = self.forward(data)
    ...
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
        data:       collated feature dict from features["feature"].data
        hist_steps: optional history length including present index.
                    If None, inferred from target shape or valid_mask.

    Returns:
        normalized data dict (in-place modifications + returned)
    """
    assert "origin" in data, (
        "data['origin'] missing — cache was not built with first_time=True"
    )
    assert "angle" in data, (
        "data['angle'] missing — cache was not built with first_time=True"
    )

    # ------------------------------------------------------------------
    # Skip if already normalized — check UTM origin magnitude, not data
    # values, to avoid false negatives near map origin
    # ------------------------------------------------------------------
    center_xy = data["origin"].float()    # (B, 2)
    center_angle = data["angle"].float()  # (B,)

    if center_xy.abs().max().item() < 100.0:
        # origin is near zero → already in ego-relative frame
        return data

    B = center_xy.shape[0]

    cos_a = torch.cos(center_angle)
    sin_a = torch.sin(center_angle)

    # Rotation matrix (B, 2, 2)
    # Matches PlutoFeature.normalize numpy convention:
    #   rotate_mat = [[cos, -sin], [sin, cos]]
    #   rotated = (pos - center_xy) @ rotate_mat
    R = torch.stack(
        [
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a,  cos_a], dim=-1),
        ],
        dim=1,
    )  # (B, 2, 2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def rot_pos(xy: torch.Tensor) -> torch.Tensor:
        """Subtract UTM center and rotate. xy: (B, ..., 2)"""
        shape = xy.shape
        xy_f = xy.float().reshape(B, -1, 2)
        xy_c = xy_f - center_xy.unsqueeze(1)
        xy_r = torch.bmm(xy_c, R.transpose(1, 2))
        return xy_r.reshape(shape)

    def rot_vec(xy: torch.Tensor) -> torch.Tensor:
        """Rotate without subtracting center (for vectors). xy: (B, ..., 2)"""
        shape = xy.shape
        xy_f = xy.float().reshape(B, -1, 2)
        xy_r = torch.bmm(xy_f, R.transpose(1, 2))
        return xy_r.reshape(shape)

    def rot_angle(a: torch.Tensor) -> torch.Tensor:
        """Subtract center_angle and wrap to [-pi, pi]. a: (B, ...)"""
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

    if "acceleration" in agent:
        agent["acceleration"] = rot_vec(agent["acceleration"].clone())

    # ------------------------------------------------------------------
    # Infer hist_steps safely
    # ------------------------------------------------------------------
    if hist_steps is None:
        if "target" in agent:
            T_total = agent["position"].shape[2]
            T_future = agent["target"].shape[2]
            hist_steps = T_total - T_future
        else:
            # fallback: PLUTO default 2s history @ 0.1s + present frame
            hist_steps = 21

    # Recompute target now that position + heading are in ego-relative frame
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
            pc = mp["polygon_center"].float().clone()  # (B, M, 3)
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
    # Current state — zero out xy and heading (matches normalize())
    # ------------------------------------------------------------------
    if "current_state" in data:
        cs = data["current_state"].float().clone()
        cs[:, :3] = 0.0
        data["current_state"] = cs

    return data


def _recompute_target(data: dict, hist_steps: int) -> None:
    """
    Recompute data["agent"]["target"] from already-normalized
    position and heading.

    Matches PlutoFeature.normalize() exactly:
        target_position = position[:, hist_steps:] - position[:, hist_steps-1, None]
        target_heading  = heading[:, hist_steps:]  - heading[:, hist_steps-1, None]
        target = concat([target_position, target_heading], dim=-1)
        target[~valid_mask] = 0

    Output shape: (B, A, T_future, 3)
    """
    pos = data["agent"]["position"].float()    # (B, A, T_total, 2)
    hdg = data["agent"]["heading"].float()     # (B, A, T_total)
    vmask = data["agent"]["valid_mask"]        # (B, A, T_total)

    T_total = pos.shape[2]
    hist_steps = min(hist_steps, T_total - 1)

    origin_pos = pos[:, :, hist_steps - 1, :]    # (B, A, 2)
    origin_hdg = hdg[:, :, hist_steps - 1]       # (B, A)

    fut_pos = pos[:, :, hist_steps:, :]          # (B, A, T_future, 2)
    fut_hdg = hdg[:, :, hist_steps:]             # (B, A, T_future)

    tgt_pos = fut_pos - origin_pos.unsqueeze(2)               # (B, A, T_future, 2)
    tgt_hdg = wrap_to_pi(fut_hdg - origin_hdg.unsqueeze(2))  # (B, A, T_future)

    target = torch.cat(
        [tgt_pos, tgt_hdg.unsqueeze(-1)],
        dim=-1,
    )  # (B, A, T_future, 3)

    # Zero out invalid future timesteps
    fut_mask = vmask[:, :, hist_steps:]          # (B, A, T_future)
    target = target * fut_mask.unsqueeze(-1).float()

    data["agent"]["target"] = target
