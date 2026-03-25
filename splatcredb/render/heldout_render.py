"""Held-out rendering utilities.

This module uses a deterministic proxy renderer for v0.1.
It is not a physical 3DGS rasterizer, but it preserves the benchmark protocol:
pruning the Gaussian set changes the rendered held-out predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HeldoutRenderOutput:
    """Container for proxy held-out rendering outputs."""

    predictions: np.ndarray  # [V, H, W]
    view_weights: np.ndarray  # [V]


def _resolve_target_shape(cameras: dict, target_shape: tuple[int, int] | None = None) -> tuple[int, int, int]:
    num_views = int(cameras.get("num_views", 2))
    if target_shape is not None:
        return num_views, int(target_shape[0]), int(target_shape[1])
    height = int(cameras.get("height", 4))
    width = int(cameras.get("width", 4))
    return num_views, height, width


def render_heldout_views(
    positions: np.ndarray,
    opacity: np.ndarray,
    cameras: dict,
    keep_mask: np.ndarray | None = None,
    target_shape: tuple[int, int] | None = None,
) -> HeldoutRenderOutput:
    """Render held-out predictions with a deterministic proxy.

    Proxy logic:
    - Active Gaussian set is selected by ``keep_mask``.
    - Per-view outputs are generated from weighted aggregates of Gaussian stats.
    - Fewer/changed gaussians produce different view tensors.
    """
    positions = np.asarray(positions, dtype=float)
    opacity = np.asarray(opacity, dtype=float).reshape(-1)
    n = positions.shape[0]

    if keep_mask is None:
        keep_mask = np.ones((n,), dtype=bool)
    keep_mask = np.asarray(keep_mask, dtype=bool)

    active_pos = positions[keep_mask]
    active_op = opacity[keep_mask]

    views, h, w = _resolve_target_shape(cameras, target_shape=target_shape)
    grid_y, grid_x = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")

    if active_pos.size == 0:
        preds = np.zeros((views, h, w), dtype=float)
        return HeldoutRenderOutput(predictions=preds, view_weights=np.zeros((views,), dtype=float))

    centroid = active_pos.mean(axis=0)
    spread = np.std(active_pos, axis=0) + 1e-6
    op_mean = float(np.mean(active_op))

    view_weights = np.linspace(0.8, 1.2, views, dtype=float)
    preds = np.zeros((views, h, w), dtype=float)

    # Simple structured field, dependent on scene stats and view index.
    for v in range(views):
        phase = view_weights[v] * (centroid[0] + 0.5 * centroid[1] + 0.25 * centroid[2])
        fx = np.cos((grid_x + phase) * np.pi) * spread[0]
        fy = np.sin((grid_y + phase) * np.pi) * spread[1]
        field = op_mean + 0.5 * (fx + fy)
        preds[v] = np.clip(field, 0.0, 1.0)

    return HeldoutRenderOutput(predictions=preds, view_weights=view_weights)
