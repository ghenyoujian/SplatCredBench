import numpy as np

def reprojection_confidence(reproj_feature: np.ndarray) -> np.ndarray:
    f=np.asarray(reproj_feature,dtype=float)
    return 1.0/(1.0+f)
