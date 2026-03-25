from pathlib import Path

def load_realestate_like(path: str | Path) -> dict:
    return {"dataset": "realestate_like", "path": str(Path(path))}
