import numpy as np

def action_to_onehot(action, nonspatial_size, spatial_size):
    action_arr = np.zeros((1, nonspatial_size + 4*spatial_size))
    spatial, nonspatial = action
    action_arr[0,nonspatial] = 1
    for k in range(len(spatial)):
        for l in range(len(spatial[k])):
            action_arr[0,nonspatial_size + (l+2*k)*spatial_size + spatial[k][l]] = 1
    return action_arr
