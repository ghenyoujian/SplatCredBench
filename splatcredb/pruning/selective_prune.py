import numpy as np

def selective_prune(confidence: np.ndarray, max_prune_ratio: float=0.3) -> np.ndarray:
    conf=np.asarray(confidence)
    n=conf.shape[0]
    prune=int(np.clip(max_prune_ratio,0.0,1.0)*n)
    idx=np.argsort(conf)
    keep=np.ones(n,dtype=bool)
    keep[idx[:prune]]=False
    return keep
