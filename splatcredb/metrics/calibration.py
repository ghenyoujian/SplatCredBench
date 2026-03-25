"""Calibration metrics for deterministic placeholder confidence."""

from __future__ import annotations

import numpy as np


def compute_calibration_metrics(confidence: np.ndarray, oracle_error: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Compute Gaussian-confidence calibration metrics.

    Returns:
      - g_ece: binned expected calibration error using utility target 1/(1+error)
      - g_brier: mean squared confidence error against that utility target
      - g_mce: max calibration gap (kept for debugging)
    """
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    err = np.asarray(oracle_error, dtype=float).reshape(-1)
    n = min(conf.size, err.size)
    if n == 0:
        return {"g_ece": 0.0, "g_brier": 0.0, "g_mce": 0.0}

    conf = conf[:n]
    target = 1.0 / (1.0 + err[:n])
    g_brier = float(np.mean((conf - target) ** 2))

    edges = np.linspace(0.0, 1.0, max(2, bins + 1))
    ece = 0.0
    mce = 0.0
    for i in range(edges.size - 1):
        left, right = edges[i], edges[i + 1]
        mask = (conf >= left) & (conf < right if i < edges.size - 2 else conf <= right)
        if not np.any(mask):
            continue
        gap = abs(float(np.mean(conf[mask]) - np.mean(target[mask])))
        ece += gap * (float(np.sum(mask)) / n)
        mce = max(mce, gap)

    return {"g_ece": float(ece), "g_brier": g_brier, "g_mce": float(mce)}
