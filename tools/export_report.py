from __future__ import annotations
import argparse, json
from pathlib import Path
from splatcredb.report import export_html_report
from splatcredb.utils.paths import ensure_dir

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Export an HTML report from metric JSON.')
    p.add_argument('--input-json', required=True)
    p.add_argument('--report-dir', required=True)
    return p.parse_args()

def main() -> None:
    args=parse_args()
    data=json.loads(Path(args.input_json).read_text()) if Path(args.input_json).exists() else {'warning':'input-json not found'}
    out=export_html_report(data, ensure_dir(args.report_dir)/'report.html')
    print(f'Saved report: {out}')

if __name__=='__main__':
    main()
