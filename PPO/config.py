"""

Put relevant constants in this file

"""
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES, FeatureType
CATEGORICAL = FeatureType.CATEGORICAL
SCALAR = FeatureType.SCALAR

class DeepMind2017Config():
    screen_shape = (1968,84,84)
    minimap_shape = (30,84,84)
    nonspatial_size = 11
    FILTERS1 = 16
    FILTERS2 = 32
    FC1SIZE = 256
    FC2SIZE = 256
    latent_size = 50
    

