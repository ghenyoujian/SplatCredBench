from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, yaml
from splatcredb.baselines import density_confidence, hybrid_confidence, learned_confidence, opacity_confidence, reprojection_confidence
from splatcredb.features import compute_density_features, compute_gaussian_features, compute_geometry_features, compute_reprojection_features
from splatcredb.io import load_cameras, load_npz_scene, load_ply_scene
from splatcredb.metrics import compute_calibration_metrics, compute_detection_metrics, compute_sru_auc, summarize_metrics
from splatcredb.oracle import compute_geometry_error, compute_oracle_error, compute_topology_error
from splatcredb.utils.paths import ensure_dir

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Evaluate placeholder confidence baselines for 3DGS scenes.')
    p.add_argument('--scene', required=True)
    p.add_argument('--cameras', required=True)
    p.add_argument('--targets', default='')
    p.add_argument('--baseline', choices=['opacity','density','reproj','hybrid','learned'], default='hybrid')
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--report-dir', default='outputs/eval')
    return p.parse_args()

def _load_scene(path: str):
    return load_npz_scene(path) if path.endswith('.npz') else load_ply_scene(path)

def main() -> None:
    args=parse_args()
    cfgp=Path(args.config)
    cfg=yaml.safe_load(cfgp.read_text()) if cfgp.exists() else {}
    scene=_load_scene(args.scene)
    cams=load_cameras(args.cameras)
    gfeat=compute_gaussian_features(scene.positions, scene.opacity)
    density=compute_density_features(scene.positions)
    reproj=compute_reprojection_features(scene.positions, int(cams.get('num_views',2)))
    _=compute_geometry_features(scene.positions)
    if args.baseline=='opacity':
        conf=opacity_confidence(gfeat['opacity'])
    elif args.baseline=='density':
        conf=density_confidence(density)
    elif args.baseline=='reproj':
        conf=reprojection_confidence(reproj)
    elif args.baseline=='learned':
        conf=learned_confidence(np.stack([gfeat['opacity'], density, reproj], axis=1))
    else:
        conf=hybrid_confidence(opacity_confidence(gfeat['opacity']), density_confidence(density), reprojection_confidence(reproj))
    geom_err=compute_geometry_error(scene.positions)
    topo_err=compute_topology_error(scene.positions)
    render_err=np.abs(1.0-conf)
    oracle_err=compute_oracle_error(render_err, geom_err, topo_err)
    sru=compute_sru_auc(conf, oracle_err, steps=int(cfg.get('metrics',{}).get('sru_auc',{}).get('retention_steps',20)))
    det=compute_detection_metrics(conf, oracle_err)
    cal=compute_calibration_metrics(conf, oracle_err, bins=int(cfg.get('metrics',{}).get('calibration',{}).get('bins',10)))
    summary=summarize_metrics(sru, det, cal)
    summary.update({'baseline': args.baseline, 'scene': args.scene, 'targets': args.targets})
    outdir=ensure_dir(args.report_dir)
    out=outdir/'summary.json'; out.write_text(json.dumps(summary, indent=2))
    print('=== SplatCredBench Evaluation Summary ===')
    print(f'Scene: {args.scene}')
    print(f'Baseline: {args.baseline}')
    for k,v in summary.items():
        if isinstance(v,float): print(f'{k}: {v:.6f}')
    print(f'Saved summary: {out}')

if __name__=='__main__':
    main()
