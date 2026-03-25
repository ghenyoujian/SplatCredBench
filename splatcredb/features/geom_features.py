import numpy as np

def compute_geometry_features(positions: np.ndarray) -> dict[str, np.ndarray]:
    return {"x": positions[:,0], "y": positions[:,1], "z": positions[:,2]}
