import numpy as np

def learned_confidence(features: np.ndarray) -> np.ndarray:
    if features.ndim==1:
        features=features[:,None]
    return np.clip(features.mean(axis=1),0.0,1.0)
