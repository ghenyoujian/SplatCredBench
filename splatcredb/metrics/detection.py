"""Detection metrics (simplified deterministic approximations)."""

from __future__ import annotations

import numpy as np


def _rank_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    wins = 0.0
    ties = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        ties += float(np.sum(p == neg))
    return float((wins + 0.5 * ties) / (pos.size * neg.size))


def compute_detection_metrics(confidence: np.ndarray, oracle_error: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute GED-AUROC/AUPR style placeholders from binary error labels."""
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    err = np.asarray(oracle_error, dtype=float).reshape(-1)
    n = min(conf.size, err.size)
    if n == 0:
        return {"ged_auroc": 0.5, "ged_aupr": 0.0}

    labels = (err[:n] >= threshold).astype(int)
    anomaly_score = 1.0 - conf[:n]
    auroc = _rank_auroc(labels, anomaly_score)

    order = np.argsort(-anomaly_score)
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    denom = max(1, int(np.sum(y)))
    recall = tp / denom
    aupr = float(np.trapezoid(precision, recall)) if y.size > 1 else float(precision[0])

    return {"ged_auroc": float(auroc), "ged_aupr": float(max(0.0, min(1.0, aupr)))}
