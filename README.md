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


## Roadmap

- Implement robust 3DGS scene readers for multiple feed-forward formats.
- Add oracle error backends from held-out rendering and geometry diagnostics.
- Expand confidence baselines and include learned confidence models.
- Standardize benchmark protocols and reporting templates.

