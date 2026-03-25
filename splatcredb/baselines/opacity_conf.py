"""Opacity confidence baseline."""

from __future__ import annotations

import numpy as np


def opacity_confidence(opacity: np.ndarray) -> np.ndarray:
    """Use min-max normalized opacity in [0, 1] as confidence."""
    values = np.asarray(opacity, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.full(values.shape, 0.5, dtype=float)
    return (values - lo) / (hi - lo)
