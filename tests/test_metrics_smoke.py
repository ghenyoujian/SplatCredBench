"""Smoke tests for rendering-driven SRU pipeline."""

import numpy as np

from splatcredb.metrics import compute_sru_auc


def test_sru_curves_and_auc_exist() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.3, 0.2, 0.1],
            [0.4, 0.3, 0.2],
        ],
        dtype=float,
    )
    opacity = np.array([0.2, 0.4, 0.7, 0.9], dtype=float)
    confidence = np.array([0.1, 0.3, 0.8, 0.9], dtype=float)
    cameras = {"num_views": 2, "height": 4, "width": 4}
    targets = np.full((2, 4, 4), 0.5, dtype=float)

    out = compute_sru_auc(positions, opacity, confidence, cameras, targets)

    assert "sru_auc_psnr" in out
    assert "sru_auc_ssim" in out
    assert "sru_auc_lpips" in out
    assert "sru_auc_norm" in out
    curves = out["curves"]
    assert len(curves["prune_ratios"]) == len(curves["psnr"]) == len(curves["ssim"]) == len(curves["lpips"])
