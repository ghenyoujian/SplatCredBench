import numpy as np

def compute_detection_metrics(confidence: np.ndarray, oracle_error: np.ndarray, threshold: float=0.5) -> dict[str,float]:
    conf=np.asarray(confidence)
    err=np.asarray(oracle_error)
    y=(err>=threshold).astype(float)
    s=1.0-conf
    corr=float(np.corrcoef(s,y)[0,1]) if s.size>1 else 0.0
    aupr=float((s*y).sum()/max(1.0,y.sum()))
    return {"ged_auroc": (np.clip(corr,-1.0,1.0)+1.0)/2.0, "ged_aupr": aupr}
