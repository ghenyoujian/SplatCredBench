import numpy as np

def compute_reprojection_features(positions: np.ndarray, num_views: int=2) -> np.ndarray:
    return np.linalg.norm(positions, axis=1)/max(1,num_views)
