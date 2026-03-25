import numpy as np
from splatcredb.metrics import compute_calibration_metrics, compute_detection_metrics, compute_sru_auc, summarize_metrics

def test_metrics_on_toy_arrays() -> None:
    c=np.array([0.9,0.7,0.2,0.1]); e=np.array([0.1,0.3,0.8,0.9])
    sru=compute_sru_auc(c,e); det=compute_detection_metrics(c,e,threshold=0.5); cal=compute_calibration_metrics(c,e); out=summarize_metrics(sru,det,cal)
    assert "sru_auc" in out and "ged_auroc" in out and "ece" in out
