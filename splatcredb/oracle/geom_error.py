import numpy as np

def compute_geometry_error(positions: np.ndarray) -> np.ndarray:
    c=positions.mean(axis=0, keepdims=True)
    return np.linalg.norm(positions-c, axis=1)
