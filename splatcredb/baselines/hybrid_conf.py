import numpy as np

def hybrid_confidence(opacity: np.ndarray, density: np.ndarray, reproj: np.ndarray, weights: tuple[float,float,float]=(0.4,0.3,0.3)) -> np.ndarray:
    w0,w1,w2=weights
    return np.clip(w0*opacity+w1*density+w2*reproj,0.0,1.0)
