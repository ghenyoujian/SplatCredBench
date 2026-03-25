from pathlib import Path

def load_scannet_like(path: str | Path) -> dict:
    return {"dataset": "scannet_like", "path": str(Path(path))}
