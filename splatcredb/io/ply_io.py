from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class GaussianScene:
    positions: np.ndarray
    opacity: np.ndarray
    metadata: dict

def load_ply_scene(path: str | Path) -> GaussianScene:
    p=Path(path)
    n=16
    positions=np.linspace(0.0,1.0,n*3).reshape(n,3)
    opacity=np.linspace(0.1,1.0,n)
    return GaussianScene(positions=positions, opacity=opacity, metadata={"source": str(p), "format": "ply"})
