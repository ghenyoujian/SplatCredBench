"""CLI: deterministic selective pruning artifact generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from splatcredb.baselines import opacity_confidence
from splatcredb.io import load_npz_scene, load_ply_scene
from splatcredb.pruning import selective_prune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune low-confidence gaussians (deterministic placeholder).")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--confidence", default="", help="Optional summary.json path to reuse confidence_mean")
    parser.add_argument("--max-prune-ratio", type=float, default=0.3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load_scene(path: str):
    return load_npz_scene(path) if str(path).endswith(".npz") else load_ply_scene(path)


def main() -> None:
    args = parse_args()
    scene = _load_scene(args.scene)
    base_conf = opacity_confidence(np.asarray(scene.opacity, dtype=float))
    keep_mask = selective_prune(base_conf, max_prune_ratio=args.max_prune_ratio)

    original = int(base_conf.shape[0])
    kept = int(keep_mask.sum())
    pruned = int(original - kept)

    payload = {
        "scene": str(args.scene),
        "original_gaussians": original,
        "kept_gaussians": kept,
        "pruned_gaussians": pruned,
        "prune_ratio": float(pruned / max(1, original)),
        "confidence_mean": float(np.mean(base_conf)),
        "confidence_min": float(np.min(base_conf)),
        "confidence_max": float(np.max(base_conf)),
        "confidence_input": str(args.confidence),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
