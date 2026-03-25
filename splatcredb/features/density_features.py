"""Deterministic placeholder density features."""

from __future__ import annotations

import numpy as np


def compute_density_features(positions: np.ndarray) -> np.ndarray:
    """Compute inverse mean distance to all other points as density proxy."""
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=float)

    dmat = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    mean_dist = dmat.mean(axis=1)
    density = 1.0 / (1.0 + mean_dist)
    lo, hi = float(density.min()), float(density.max())
    if hi - lo < 1e-12:
        return np.full((n,), 0.5, dtype=float)
    return (density - lo) / (hi - lo)
