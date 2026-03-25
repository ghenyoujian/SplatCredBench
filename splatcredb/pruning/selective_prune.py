"""Deterministic confidence-based selective pruning."""

from __future__ import annotations

import numpy as np


def selective_prune(confidence: np.ndarray, max_prune_ratio: float = 0.3) -> np.ndarray:
    """Return keep-mask by removing the least confident entries."""
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    n = conf.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)

    prune_ratio = float(np.clip(max_prune_ratio, 0.0, 1.0))
    prune_count = int(np.floor(prune_ratio * n))
    order = np.argsort(conf)  # low confidence first

    keep = np.ones(n, dtype=bool)
    if prune_count > 0:
        keep[order[:prune_count]] = False
    return keep
