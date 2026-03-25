"""CLI smoke tests for SRU protocol workflow."""

import json
from pathlib import Path

from tools import evaluate, export_report, prune_and_render


TOY_SCENE = "examples/scene_001/scene.npz"  # optional file; loader has deterministic fallback.
TOY_CAMERAS = "examples/scene_001/cameras.json"
TOY_TARGETS = "examples/scene_001/targets.npy"  # optional file; evaluator has deterministic fallback.


def test_cli_mains_run(tmp_path: Path, monkeypatch) -> None:
    eval_dir = tmp_path / "eval"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--scene",
            TOY_SCENE,
            "--cameras",
            TOY_CAMERAS,
            "--targets",
            TOY_TARGETS,
            "--baseline",
            "hybrid",
            "--report-dir",
            str(eval_dir),
        ],
    )
    evaluate.main()

    summary_json = eval_dir / "summary.json"
    assert summary_json.exists()
    summary = json.loads(summary_json.read_text())

    for key in ["sru_auc_psnr", "sru_auc_ssim", "sru_auc_lpips", "sru_auc_norm", "curves"]:
        assert key in summary
    for key in ["prune_ratios", "psnr", "ssim", "lpips"]:
        assert key in summary["curves"]

    report_dir = tmp_path / "report"
    monkeypatch.setattr("sys.argv", ["export_report", "--input-json", str(summary_json), "--report-dir", str(report_dir)])
    export_report.main()

    prune_out = tmp_path / "prune" / "pruned.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune",
            "--scene",
            TOY_SCENE,
            "--confidence",
            str(summary_json),
            "--max-prune-ratio",
            "0.25",
            "--output",
            str(prune_out),
        ],
    )
    prune_and_render.main()

    assert (report_dir / "report.html").exists()
    assert (report_dir / "assets" / "curve_psnr.png").exists()
    assert (report_dir / "assets" / "curve_ssim.png").exists()
    assert (report_dir / "assets" / "curve_lpips.png").exists()
    assert (report_dir / "assets" / "curve_gaussian_counts.png").exists()
    assert prune_out.exists()
