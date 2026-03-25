from pathlib import Path

def export_html_report(metrics: dict, output_path: str | Path, title: str="SplatCredBench Report") -> Path:
    p=Path(output_path); p.parent.mkdir(parents=True, exist_ok=True)
    rows="\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in metrics.items())
    p.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head><body><h1>{title}</h1><table border='1' cellspacing='0' cellpadding='6'><tr><th>Metric</th><th>Value</th></tr>{rows}</table></body></html>")
    return p
