# SplatCredBench

**A Confidence Benchmark for Feed-Forward 3D Gaussian Splatting.**

SplatCredBench is a small, research-friendly v0.1 prototype for benchmarking per-Gaussian confidence in feed-forward 3DGS pipelines.

## Motivation

This repository provides a practical, deterministic end-to-end loop for:
- confidence estimation baselines,
- selective utility (SRU-AUC),
- error detection proxies (GED-AUROC/AUPR),
- calibration checks.

Current logic is intentionally simplified and deterministic; it is a scaffold for iterative benchmark development, not the final research implementation.

## Quickstart (v0.1)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run the toy demo

Evaluate toy scene:

```bash
splatcred-eval \
  --scene examples/scene_001/scene.npz \
  --cameras examples/scene_001/cameras.json \
  --targets examples/scene_001/targets.npy \
  --baseline hybrid \
  --config configs/default.yaml \
  --report-dir outputs/eval_demo
```

Export HTML report:

```bash
splatcred-report \
  --input-json outputs/eval_demo/summary.json \
  --report-dir outputs/report_demo
```

Run deterministic prune artifact:

```bash
splatcred-prune \
  --scene examples/scene_001/scene.npz \
  --max-prune-ratio 0.3 \
  --output outputs/prune_demo/pruned.json
```

## Expected output files

- `outputs/eval_demo/summary.json`
- `outputs/report_demo/report.html`
- `outputs/prune_demo/pruned.json`

## Repository structure

- `splatcredb/`: package code (I/O, features, baselines, metrics, pruning, report).
- `tools/`: CLI entrypoints.
- `configs/`: default and baseline YAML configs.
- `examples/scene_001/`: deterministic toy data for end-to-end runs.
- `tests/`: smoke tests for imports, metrics, and CLIs.

## Roadmap

- Replace placeholder formulas with benchmark-grade implementations.
- Add richer held-out protocols and diagnostics.
- Extend reporting and standardized experiment configs.

## Citation

```bibtex
@misc{splatcredb2026,
  title={SplatCredBench: A Confidence Benchmark for Feed-Forward 3D Gaussian Splatting},
  author={TODO},
  year={2026},
  note={GitHub repository}
}
```
