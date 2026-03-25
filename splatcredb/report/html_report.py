"""HTML report exporter with embedded SRU curve plots."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _plot_curve(x: list[float], y: list[float], ylabel: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Prune ratio")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def export_html_report(metrics: dict[str, Any], output_path: str | Path, title: str = "SplatCredBench Report") -> Path:
    """Export static HTML summary report with SRU plots."""
    output = Path(output_path)
    report_dir = output.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    scene = metrics.get("scene", "unknown")
    baseline = metrics.get("baseline", "unknown")
    curves = metrics.get("curves", {})
    prune_ratios = [float(v) for v in curves.get("prune_ratios", [])]

    scalar_items = {
        k: v
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and k not in {"num_gaussians"}
    }
    rows = "\n".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(_fmt(value))}</td></tr>" for key, value in scalar_items.items()
    )

    asset_paths: dict[str, Path] = {}
    if prune_ratios:
        plot_dir = report_dir / "assets"
        for key, label in [
            ("psnr", "PSNR"),
            ("ssim", "SSIM"),
            ("lpips", "LPIPS (proxy)"),
            ("gaussian_counts", "Gaussian Count"),
        ]:
            y = [float(v) for v in curves.get(key, [])]
            if y:
                asset_paths[key] = _plot_curve(prune_ratios, y, label, plot_dir / f"curve_{key}.png")

    image_blocks = []
    for key in ["psnr", "ssim", "lpips", "gaussian_counts"]:
        if key in asset_paths:
            rel = asset_paths[key].relative_to(report_dir)
            image_blocks.append(f"<h3>{escape(key)}</h3><img src=\"{escape(str(rel))}\" alt=\"{escape(key)} curve\" width=\"520\" />")

    images_html = "\n".join(image_blocks) if image_blocks else "<p>No curve arrays found in summary.json.</p>"

    html = f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>{escape(title)}</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 2rem; }}
      table {{ border-collapse: collapse; width: 100%; max-width: 720px; }}
      th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
      th {{ background: #f5f5f5; }}
      code {{ background: #f3f3f3; padding: 0.1rem 0.3rem; }}
      img {{ border: 1px solid #ddd; margin-bottom: 1rem; }}
    </style>
  </head>
  <body>
    <h1>{escape(title)}</h1>
    <p><strong>Scene:</strong> <code>{escape(str(scene))}</code></p>
    <p><strong>Baseline:</strong> <code>{escape(str(baseline))}</code></p>

    <h2>Scalar Metrics</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      {rows}
    </table>

    <h2>SRU Curves</h2>
    {images_html}

    <h2>Notes</h2>
    <p>Held-out rendering is a deterministic proxy in v0.1 and is designed for protocol wiring, not final visual fidelity.</p>
  </body>
</html>
"""
    output.write_text(html)
    return output
