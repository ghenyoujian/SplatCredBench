from pathlib import Path
import json

def save_metric_tables(metrics: dict, output_path: str | Path) -> Path:
    p=Path(output_path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(metrics, indent=2)); return p
