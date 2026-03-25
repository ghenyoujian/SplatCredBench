"""Density confidence baseline."""

from __future__ import annotations

import numpy as np


def density_confidence(density_feature: np.ndarray) -> np.ndarray:
    """Use normalized density proxy directly as confidence."""
    values = np.asarray(density_feature, dtype=float).reshape(-1)
    return np.clip(values, 0.0, 1.0)
