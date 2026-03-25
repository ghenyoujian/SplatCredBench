def summarize_metrics(sru_auc: float, detection: dict[str,float], calibration: dict[str,float]) -> dict[str,float]:
    out={"sru_auc": float(sru_auc)}
    out.update(detection)
    out.update(calibration)
    return out
