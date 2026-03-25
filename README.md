# SplatCredBench

**A Confidence Benchmark for Feed-Forward 3D Gaussian Splatting.**

SplatCredBench is a lightweight, research-oriented benchmark scaffold for studying confidence estimation in feed-forward 3D Gaussian Splatting (3DGS). The repository is intentionally minimal and ready for iterative development.

## Motivation

Feed-forward 3DGS pipelines can reconstruct scenes quickly, but confidence and reliability estimates are often missing or inconsistent. This project provides a practical benchmark structure to compare confidence signals, evaluate selective behavior, and report calibration quality.

## Core ideas

- **Gaussian confidence estimation**: estimate per-Gaussian confidence scores from geometric, density, and reprojection cues.
- **SRU-AUC**: selective reconstruction utility under pruning/retention policies.
- **GED-AUROC/AUPR**: detection quality for error-prone Gaussian regions.
- **Calibration**: assess confidence alignment with oracle reconstruction error.

## Repository structure

- `splatcredb/`: core package with I/O, features, baselines, metrics, pruning, rendering, and reporting placeholders.
- `tools/`: lightweight CLI entry points.
- `configs/`: default and baseline configs.
- `datasets/`: dataset loader/protocol placeholders.
- `examples/`: minimal example layout.
- `tests/`: smoke tests for imports, metrics, and CLIs.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Minimal usage

```bash
splatcred-eval --scene examples/scene_001/scene.npz --cameras examples/scene_001/cameras.json --targets examples/scene_001/targets.npy --baseline hybrid --config configs/default.yaml --report-dir outputs/eval_demo
```

```bash
splatcred-prune --scene examples/scene_001/scene.npz --confidence outputs/eval_demo/summary.json --max-prune-ratio 0.3 --output outputs/pruned_scene.json
```

```bash
splatcred-report --input-json outputs/eval_demo/summary.json --report-dir outputs/report_demo
```

## Roadmap

- Implement robust 3DGS scene readers for multiple feed-forward formats.
- Add oracle error backends from held-out rendering and geometry diagnostics.
- Expand confidence baselines and include learned confidence models.
- Standardize benchmark protocols and reporting templates.

## Citation

If you use this repository, please cite:

```bibtex
@misc{splatcredb2026,
  title={SplatCredBench: A Confidence Benchmark for Feed-Forward 3D Gaussian Splatting},
  author={TODO},
  year={2026},
  note={GitHub repository}
}
```
