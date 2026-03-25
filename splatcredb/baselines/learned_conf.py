"""Pseudo learned confidence baseline for deterministic v0.1 runs."""

from __future__ import annotations

import numpy as np


def learned_confidence(features: np.ndarray) -> np.ndarray:
    """Return a deterministic linear blend of input features.

    TODO: replace with model inference in future versions.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features[:, None]
    weights = np.linspace(1.0, 0.5, features.shape[1], dtype=float)
    score = (features * weights[None, :]).sum(axis=1) / weights.sum()
    return np.clip(score, 0.0, 1.0)
