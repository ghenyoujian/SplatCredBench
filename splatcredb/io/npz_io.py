from pathlib import Path
import numpy as np
from .ply_io import GaussianScene

def load_npz_scene(path: str | Path) -> GaussianScene:
    p=Path(path)
    if p.exists() and p.suffix=='.npz':
        with np.load(p, allow_pickle=True) as data:
            positions=data.get('positions', np.zeros((16,3)))
            opacity=data.get('opacity', np.full((positions.shape[0],),0.5))
    else:
        positions=np.zeros((16,3))
        opacity=np.full((16,),0.5)
    return GaussianScene(positions=positions, opacity=opacity, metadata={"source": str(p), "format": "npz"})
