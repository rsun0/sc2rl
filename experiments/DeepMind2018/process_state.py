from pysc2.actions import FUNCTIONS
import numpy as np

ACTION_SPACE = len(FUNCTIONS)

def state_processor(obs):

    minimap = obs.observation.feature_minimap
    screen = obs.observation.feature_screen
    player = obs.observation.player

    avail_actions = np.zeros(ACTION_SPACE)
    avail_actions[obs.observation.available_actions] = 1

    info = None

    return (minimap, screen, player), avail_actions, info
