"""Rendering-driven SRU curves and AUC metrics for v0.1."""

from __future__ import annotations

import numpy as np

from splatcredb.pruning import selective_prune
from splatcredb.render.heldout_render import render_heldout_views


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 80.0
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_proxy(pred: np.ndarray, target: np.ndarray) -> float:
    """Global SSIM-style proxy (deterministic and lightweight)."""
    c1, c2 = 0.01**2, 0.03**2
    mu_x = float(np.mean(pred))
    mu_y = float(np.mean(target))
    var_x = float(np.var(pred))
    var_y = float(np.var(target))
    cov_xy = float(np.mean((pred - mu_x) * (target - mu_y)))
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(np.clip(num / max(1e-12, den), 0.0, 1.0))


def _lpips_proxy(pred: np.ndarray, target: np.ndarray) -> float:
    """LPIPS-like proxy: normalized gradient + intensity distance.

    Lower is better, matching LPIPS direction.
    """
    gx_p = np.diff(pred, axis=-1, prepend=pred[..., :1])
    gy_p = np.diff(pred, axis=-2, prepend=pred[..., :1, :])
    gx_t = np.diff(target, axis=-1, prepend=target[..., :1])
    gy_t = np.diff(target, axis=-2, prepend=target[..., :1, :])

    grad_dist = np.mean(np.abs(gx_p - gx_t) + np.abs(gy_p - gy_t))
    pix_dist = np.mean(np.abs(pred - target))
    return float(np.clip(0.5 * grad_dist + 0.5 * pix_dist, 0.0, 1.0))


def _normalize_curve(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not higher_is_better:
        arr = -arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-12:
        return np.ones_like(arr) * 0.5
    return (arr - lo) / (hi - lo)


def compute_sru_auc(
    positions: np.ndarray,
    opacity: np.ndarray,
    confidence: np.ndarray,
    cameras: dict,
    targets: np.ndarray,
    prune_ratios: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute protocol-aligned SRU curves/AUC from pruning + held-out rendering.

    Steps:
      1) sort by confidence and prune lowest confidence at each ratio
      2) render held-out predictions for active set
      3) compute PSNR/SSIM/LPIPS curves
      4) integrate AUC over prune ratio
    """
    positions = np.asarray(positions, dtype=float)
    opacity = np.asarray(opacity, dtype=float).reshape(-1)
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float)

    n = min(positions.shape[0], opacity.size, confidence.size)
    if prune_ratios is None:
        prune_ratios = np.linspace(0.0, 0.9, 10, dtype=float)
    else:
        prune_ratios = np.asarray(prune_ratios, dtype=float)

    psnr_curve: list[float] = []
    ssim_curve: list[float] = []
    lpips_curve: list[float] = []
    gaussian_counts: list[int] = []

    for ratio in prune_ratios:
        keep_mask = selective_prune(confidence[:n], max_prune_ratio=float(ratio))
        rendered = render_heldout_views(
            positions=positions[:n],
            opacity=opacity[:n],
            cameras=cameras,
            keep_mask=keep_mask,
            target_shape=(targets.shape[-2], targets.shape[-1]),
        )
        pred = rendered.predictions
        tgt = targets[: pred.shape[0]]

        psnr_curve.append(_psnr(pred, tgt))
        ssim_curve.append(_ssim_proxy(pred, tgt))
        lpips_curve.append(_lpips_proxy(pred, tgt))
        gaussian_counts.append(int(np.sum(keep_mask)))

    x = prune_ratios
    psnr_arr = np.asarray(psnr_curve, dtype=float)
    ssim_arr = np.asarray(ssim_curve, dtype=float)
    lpips_arr = np.asarray(lpips_curve, dtype=float)

    # Normalized utility combines high-better (PSNR,SSIM) and low-better (LPIPS).
    norm_psnr = _normalize_curve(psnr_arr, higher_is_better=True)
    norm_ssim = _normalize_curve(ssim_arr, higher_is_better=True)
    norm_lpips = _normalize_curve(lpips_arr, higher_is_better=False)
    norm_curve = (norm_psnr + norm_ssim + norm_lpips) / 3.0

    return {
        "sru_auc_psnr": float(np.trapezoid(psnr_arr, x)),
        "sru_auc_ssim": float(np.trapezoid(ssim_arr, x)),
        "sru_auc_lpips": float(np.trapezoid(lpips_arr, x)),
        "sru_auc_norm": float(np.trapezoid(norm_curve, x)),
        "curves": {
            "prune_ratios": [float(v) for v in x.tolist()],
            "gaussian_counts": [int(v) for v in gaussian_counts],
            "psnr": [float(v) for v in psnr_arr.tolist()],
            "ssim": [float(v) for v in ssim_arr.tolist()],
            "lpips": [float(v) for v in lpips_arr.tolist()],
            "norm": [float(v) for v in norm_curve.tolist()],
        },
    }
