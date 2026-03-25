"""Deterministic placeholder Gaussian features."""

from __future__ import annotations

import numpy as np


def compute_gaussian_features(positions: np.ndarray, opacity: np.ndarray) -> dict[str, np.ndarray]:
    """Compute simple per-Gaussian features from positions and opacity.

    Returns normalized radius and raw opacity as a minimal interpretable feature set.
    """
    positions = np.asarray(positions, dtype=float)
    opacity = np.asarray(opacity, dtype=float).reshape(-1)
    radius = np.linalg.norm(positions, axis=1)
    rmax = float(np.max(radius)) if radius.size else 1.0
    radius_norm = radius / max(1e-8, rmax)
    return {"radius": radius_norm, "opacity": opacity}
