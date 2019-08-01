import sys
sys.path.insert(0, "../../interface/")

from pysc2.env import sc2_env
from pysc2.lib import features, protocol
from process_state import state_processor
from pysc2.lib.actions import FUNCTIONS, FunctionCall
from custom_env import FullStateActionEnvironment

import numpy as np

env = FullStateActionEnvironment("DefeatRoaches")

obs, _, _, _ = env.reset()
for i in range(100):
    args = np.zeros(10)
    args[0] = 1
    args_spatial = np.zeros((3,2))
    args_spatial[0] = [10,10]
    args_spatial[2] = [50,50]
    f = [3, args, args_spatial]
    obs, _, _, _ = env.step(f)
    print(len(obs))
