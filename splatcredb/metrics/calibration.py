import numpy as np

def compute_calibration_metrics(confidence: np.ndarray, oracle_error: np.ndarray, bins: int=10) -> dict[str,float]:
    conf=np.asarray(confidence)
    tgt=1.0/(1.0+np.asarray(oracle_error))
    edges=np.linspace(0.0,1.0,bins+1)
    ece=0.0
    for i in range(bins):
        mask=(conf>=edges[i]) & ((conf<edges[i+1]) if i<bins-1 else (conf<=edges[i+1]))
        if mask.any():
            ece += abs(float(conf[mask].mean()-tgt[mask].mean()))*(mask.sum()/conf.size)
    return {"ece": float(ece), "mce": float(np.max(np.abs(conf-tgt)))}
