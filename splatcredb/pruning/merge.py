import numpy as np

def merge_gaussians(positions: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    return positions[keep_mask]
