"""CLI: deterministic confidence evaluation with rendering-driven SRU protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from splatcredb.baselines import (
    density_confidence,
    hybrid_confidence,
    learned_confidence,
    opacity_confidence,
    reprojection_confidence,
)
from splatcredb.features import (
    compute_density_features,
    compute_gaussian_features,
    compute_geometry_features,
    compute_reprojection_features,
)
from splatcredb.io import load_cameras, load_npz_scene, load_ply_scene
from splatcredb.metrics import (
    compute_calibration_metrics,
    compute_detection_metrics,
    compute_sru_auc,
)
from splatcredb.oracle import (
    compute_geometry_error,
    compute_oracle_error,
    compute_render_error,
    compute_topology_error,
)
from splatcredb.utils.paths import ensure_dir


BASELINES = {
    "opacity": lambda o, d, r: opacity_confidence(o),
    "density": lambda o, d, r: density_confidence(d),
    "reproj": lambda o, d, r: reprojection_confidence(r),
    "hybrid": lambda o, d, r: hybrid_confidence(opacity_confidence(o), density_confidence(d), reprojection_confidence(r)),
    "learned": lambda o, d, r: learned_confidence(np.stack([opacity_confidence(o), density_confidence(d), reprojection_confidence(r)], axis=1)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate confidence baseline with SRU held-out pruning protocol.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--targets", default="")
    parser.add_argument("--baseline", choices=sorted(BASELINES.keys()), default="hybrid")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--report-dir", default="outputs/eval")
    return parser.parse_args()


def _load_scene(path: str):
    return load_npz_scene(path) if str(path).endswith(".npz") else load_ply_scene(path)


def _load_targets(path: str, cameras: dict) -> np.ndarray:
    num_views = int(cameras.get("num_views", 2))
    h = int(cameras.get("height", 4))
    w = int(cameras.get("width", 4))

    if path and Path(path).exists() and path.endswith(".npy"):
        arr = np.asarray(np.load(path), dtype=float)
        if arr.ndim == 1:
            # Backward compatible vector target -> tiled images.
            base = arr.mean() if arr.size else 0.5
            tiled = np.full((num_views, h, w), float(base), dtype=float)
            return tiled
        if arr.ndim == 2:
            return np.tile(arr[None, :, :], (num_views, 1, 1))
        if arr.ndim == 3:
            return arr[:num_views]
    # deterministic fallback
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    targets = []
    for v in range(num_views):
        targets.append(np.clip(0.2 + 0.6 * (0.6 * xx + 0.4 * yy + 0.05 * v), 0.0, 1.0))
    return np.asarray(targets, dtype=float)


def _to_jsonable(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, dict):
        return {k: _to_jsonable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_jsonable(v) for v in data]
    if isinstance(data, np.ndarray):
        return [_to_jsonable(v) for v in data.tolist()]
    if isinstance(data, np.generic):
        return float(data)
    return data


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}

    scene = _load_scene(args.scene)
    cameras = load_cameras(args.cameras)
    positions = np.asarray(scene.positions, dtype=float)
    opacity = np.asarray(scene.opacity, dtype=float)

    gaussian_features = compute_gaussian_features(positions, opacity)
    density_features = compute_density_features(positions)
    reproj_features = compute_reprojection_features(positions, num_views=int(cameras.get("num_views", 2)))
    _ = compute_geometry_features(positions)

    confidence = BASELINES[args.baseline](gaussian_features["opacity"], density_features, reproj_features)

    targets = _load_targets(args.targets, cameras)
    # oracle terms retained for detection/calibration.
    render_error = compute_render_error(np.mean(targets, axis=(1, 2)), np.full((targets.shape[0],), float(1.0 - np.mean(confidence))))
    geom_error = compute_geometry_error(positions)
    topo_error = compute_topology_error(positions)
    oracle_error = compute_oracle_error(render_error, geom_error, topo_error)

    steps = int(config.get("metrics", {}).get("sru_auc", {}).get("retention_steps", 10))
    prune_ratios = np.linspace(0.0, 0.9, steps, dtype=float)
    sru = compute_sru_auc(
        positions=positions,
        opacity=opacity,
        confidence=confidence,
        cameras=cameras,
        targets=targets,
        prune_ratios=prune_ratios,
    )

    detection = compute_detection_metrics(confidence, oracle_error)
    calibration = compute_calibration_metrics(confidence, oracle_error, bins=int(config.get("metrics", {}).get("calibration", {}).get("bins", 10)))

    summary: dict[str, Any] = {
        "scene": str(args.scene),
        "baseline": args.baseline,
        "num_gaussians": int(confidence.shape[0]),
        "confidence_mean": float(np.mean(confidence)),
        "oracle_error_mean": float(np.mean(oracle_error)),
        **{k: float(v) for k, v in sru.items() if k != "curves"},
        **{k: float(v) for k, v in detection.items()},
        **{k: float(v) for k, v in calibration.items()},
        "curves": sru["curves"],
    }

    report_dir = ensure_dir(args.report_dir)
    output_json = report_dir / "summary.json"
    output_json.write_text(json.dumps(_to_jsonable(summary), indent=2))

    print("=== SplatCredBench v0.1 Evaluation ===")
    print(f"scene={args.scene}")
    print(f"baseline={args.baseline}")
    print(f"gaussians={summary['num_gaussians']}")
    print(
        f"SRU-AUC: PSNR={summary['sru_auc_psnr']:.6f} SSIM={summary['sru_auc_ssim']:.6f} "
        f"LPIPS={summary['sru_auc_lpips']:.6f} NORM={summary['sru_auc_norm']:.6f}"
    )
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
