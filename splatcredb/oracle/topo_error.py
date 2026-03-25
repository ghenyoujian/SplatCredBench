import numpy as np

def compute_topology_error(positions: np.ndarray) -> np.ndarray:
    diffs=np.diff(np.sort(positions[:,0])) if positions.shape[0]>1 else np.array([0.0])
    return np.pad(np.abs(diffs), (0, max(0, positions.shape[0]-diffs.shape[0])), mode="constant")
