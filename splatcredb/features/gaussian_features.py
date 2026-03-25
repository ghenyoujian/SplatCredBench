import numpy as np

def compute_gaussian_features(positions: np.ndarray, opacity: np.ndarray) -> dict[str, np.ndarray]:
    return {"radius": np.linalg.norm(positions, axis=1), "opacity": opacity}
