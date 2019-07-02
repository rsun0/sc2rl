import numpy as np
from config import *

def action_to_onehot(action, nonspatial_size, spatial_size):
    action_arr = np.zeros((1, nonspatial_size + 4))
    spatial, nonspatial = action
    action_arr[0,nonspatial] = 1
    for k in range(len(spatial)):
        for l in range(len(spatial[k])):
            action_arr[0,nonspatial_size + (l+2*k)] = spatial[k][l] / GraphConvConfigMinigames.spatial_width
    return action_arr
