from pathlib import Path
import json
import numpy as np

def export_with_uncertainty(path: str | Path, confidence: np.ndarray, keep_mask: np.ndarray) -> Path:
    p=Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"num_gaussians": int(confidence.shape[0]), "num_kept": int(keep_mask.sum()), "confidence_mean": float(np.mean(confidence))}, indent=2))
    return p
