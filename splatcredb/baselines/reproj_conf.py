"""Reprojection confidence baseline."""

from __future__ import annotations

import numpy as np


def reprojection_confidence(reproj_feature: np.ndarray) -> np.ndarray:
    """Turn residual proxy into confidence by inversion."""
    values = np.asarray(reproj_feature, dtype=float).reshape(-1)
    return np.clip(1.0 - values, 0.0, 1.0)
