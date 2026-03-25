from __future__ import annotations
import argparse, json
from pathlib import Path
from splatcredb.io import load_npz_scene, load_ply_scene

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Summarize potentially bad gaussians.')
    p.add_argument('--scene', required=True)
    p.add_argument('--output', default='')
    p.add_argument('--top-k', type=int, default=10)
    return p.parse_args()

def _load_scene(path: str):
    return load_npz_scene(path) if path.endswith('.npz') else load_ply_scene(path)

def main() -> None:
    args=parse_args(); scene=_load_scene(args.scene); k=min(args.top_k, scene.opacity.shape[0]); idx=scene.opacity.argsort()[:k].tolist()
    summary={'scene': args.scene, 'top_k': k, 'lowest_opacity_indices': idx}
    if args.output:
        out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(summary, indent=2)); print(f'Saved summary to {out}')
    else:
        print(json.dumps(summary, indent=2))

if __name__=='__main__':
    main()
