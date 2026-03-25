import numpy as np

def compute_oracle_error(render_err: np.ndarray, geom_err: np.ndarray, topo_err: np.ndarray) -> np.ndarray:
    m=min(render_err.shape[0], geom_err.shape[0], topo_err.shape[0])
    return (render_err[:m]+geom_err[:m]+topo_err[:m])/3.0
