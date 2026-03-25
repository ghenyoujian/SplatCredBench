import numpy as np

def trace_contributions(num_gaussians: int, num_pixels: int=16) -> np.ndarray:
    return np.full((num_gaussians,num_pixels), 1.0/max(1,num_gaussians), dtype=float)
