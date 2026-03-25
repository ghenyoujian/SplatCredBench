from __future__ import annotations
import argparse, json
from pathlib import Path
from splatcredb.baselines import opacity_confidence
from splatcredb.io import load_npz_scene, load_ply_scene
from splatcredb.pruning import export_with_uncertainty, selective_prune

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Prune low-confidence gaussians (placeholder).')
    p.add_argument('--scene', required=True)
    p.add_argument('--confidence', default='')
    p.add_argument('--max-prune-ratio', type=float, default=0.3)
    p.add_argument('--output', required=True)
    return p.parse_args()

def _load_scene(path: str):
    return load_npz_scene(path) if path.endswith('.npz') else load_ply_scene(path)

def main() -> None:
    args=parse_args()
    scene=_load_scene(args.scene)
    conf=opacity_confidence(scene.opacity)
    keep=selective_prune(conf, max_prune_ratio=args.max_prune_ratio)
    out=export_with_uncertainty(args.output, conf, keep)
    meta=Path(args.output).with_suffix('.meta.json')
    meta.write_text(json.dumps({'scene': args.scene, 'confidence_source': args.confidence, 'max_prune_ratio': args.max_prune_ratio, 'output': str(out)}, indent=2))
    print(f'Saved pruning metadata: {meta}')

if __name__=='__main__':
    main()
