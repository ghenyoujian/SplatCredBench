"""Metric summary helpers."""

from __future__ import annotations


def summarize_metrics(sru_auc: float, detection: dict[str, float], calibration: dict[str, float]) -> dict[str, float]:
    """Aggregate metric groups into plain Python float dictionary."""
    out: dict[str, float] = {"sru_auc": float(sru_auc)}
    out.update({k: float(v) for k, v in detection.items()})
    out.update({k: float(v) for k, v in calibration.items()})
    return out
