from pathlib import Path
import json

def load_cameras(path: str | Path) -> dict:
    p=Path(path)
    if p.exists() and p.suffix.lower()=='.json':
        return json.loads(p.read_text())
    return {"source": str(p), "intrinsics": [[500.0,0.0,256.0],[0.0,500.0,256.0],[0.0,0.0,1.0]], "num_views": 2}
