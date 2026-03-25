import numpy as np

def compute_render_error(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.abs(targets-predictions)
