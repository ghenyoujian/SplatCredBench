"""Hybrid confidence baseline."""

from __future__ import annotations

import numpy as np


def hybrid_confidence(
    opacity: np.ndarray,
    density: np.ndarray,
    reproj: np.ndarray,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> np.ndarray:
    """Weighted confidence blend of opacity, density, and reprojection."""
    w0, w1, w2 = weights
    blend = w0 * np.asarray(opacity) + w1 * np.asarray(density) + w2 * np.asarray(reproj)
    return np.clip(np.asarray(blend, dtype=float), 0.0, 1.0)
