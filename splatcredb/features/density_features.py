import numpy as np

def compute_density_features(positions: np.ndarray) -> np.ndarray:
    c=positions.mean(axis=0, keepdims=True)
    d=np.linalg.norm(positions-c, axis=1)
    return 1.0/(1.0+d)
