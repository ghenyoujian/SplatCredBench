import numpy as np

def opacity_confidence(opacity: np.ndarray) -> np.ndarray:
    o=np.asarray(opacity,dtype=float)
    lo,hi=float(o.min()),float(o.max())
    if hi-lo<1e-12:
        return np.full_like(o,0.5)
    return (o-lo)/(hi-lo)
