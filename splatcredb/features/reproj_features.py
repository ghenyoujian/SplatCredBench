"""Deterministic placeholder reprojection features."""

from __future__ import annotations

import numpy as np


def compute_reprojection_features(positions: np.ndarray, num_views: int = 2) -> np.ndarray:
    """Compute a stable reprojection residual proxy.

    Lower values indicate easier reprojection; output is clamped to [0, 1].
    """
    positions = np.asarray(positions, dtype=float)
    centered = positions - positions.mean(axis=0, keepdims=True)
    residual = np.linalg.norm(centered[:, :2], axis=1) / max(1, int(num_views))
    if residual.size == 0:
        return residual
    return residual / max(1e-8, float(residual.max()))
