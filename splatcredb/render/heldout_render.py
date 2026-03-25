import numpy as np

def render_heldout_views(num_views: int=2, image_size: tuple[int,int]=(32,32)) -> np.ndarray:
    h,w=image_size
    return np.zeros((num_views,h,w,3),dtype=float)
