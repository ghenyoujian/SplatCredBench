from pathlib import Path

def load_generic_scene(path: str | Path) -> dict:
    p=Path(path)
    return {"source": str(p), "num_gaussians": 16}
