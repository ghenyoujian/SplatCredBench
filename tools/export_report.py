"""CLI: export an HTML report from an evaluation summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from splatcredb.report import export_html_report
from splatcredb.utils.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a minimal HTML report from summary.json.")
    parser.add_argument("--input-json", required=True, help="Path to evaluation summary JSON")
    parser.add_argument("--report-dir", required=True, help="Directory for report artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    summary = json.loads(input_path.read_text())
    report_dir = ensure_dir(args.report_dir)
    report_path = export_html_report(summary, report_dir / "report.html")
    print(f"saved={report_path}")


if __name__ == "__main__":
    main()
