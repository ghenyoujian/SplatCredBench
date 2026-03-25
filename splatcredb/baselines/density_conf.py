import numpy as np

def density_confidence(density_feature: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(density_feature,dtype=float),0.0,1.0)
