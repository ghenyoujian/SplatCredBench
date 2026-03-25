import numpy as np

def compute_sru_auc(confidence: np.ndarray, oracle_error: np.ndarray, steps: int=20) -> float:
    c=np.asarray(confidence)
    e=np.asarray(oracle_error)
    order=np.argsort(-c)
    e=e[order]
    xs=np.linspace(0.0,1.0,max(2,steps))
    ys=[]
    n=len(e)
    for x in xs:
        k=max(1,int(n*x))
        ys.append(float(1.0/(1.0+e[:k].mean())))
    return float(np.trapz(np.array(ys), xs))
