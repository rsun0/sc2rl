from pysc2.lib.actions import FUNCTIONS
import numpy as np

ACTION_SPACE = len(FUNCTIONS)

def state_processor(obs):

    minimap = obs.observation.feature_minimap
    screen = obs.observation.feature_screen
    player = obs.observation.player

    avail_actions = np.zeros(ACTION_SPACE)
    avail_actions[obs.observation.available_actions] = 1

    minimap = minimap.reshape((1,) + minimap.shape)
    screen = screen.reshape((1,) + screen.shape)
    player = player.reshape((1,) + player.shape)

    return np.array([minimap, screen, player, avail_actions])
