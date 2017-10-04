import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_GROUP = actions.FUNCTIONS.select_control_group.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_SINGLE_SELECT = [0]
_RECALL_GROUP = [0]
_SET_GROUP = [1]

class Agent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.time = -1
        self.targeted = []

    def reset(self):
        super().reset()
        self.time = -1
        self.targeted = []

    '''
    A scripted agent designed to complete CollectMineralShards.SC2Map
    better than the scripted agent provided by DeepMind
    '''
    def step(self, obs):
        super().step(obs)
        self.time += 1
        if self.time % 2 == 0:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
            selected = obs.observation["screen"][_SELECTED]
            selected_y, selected_x = selected.nonzero()
            for m in zip(player_x, player_y):
                target = m
                for s in zip(selected_x, selected_y):
                    if s == m:
                        target = None
                        break
                if target:
                    break
            return actions.FunctionCall(_SELECT_POINT, [_SINGLE_SELECT, target])
        else:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            selected = obs.observation["screen"][_SELECTED]
            marine_y, marine_x = selected.nonzero()
            if not neutral_y.any() or not marine_y.any():
                return actions.FunctionCall(_NO_OP, [])
            marine = marine_x[0], marine_y[0]
            closest = None
            min_dist = None
            for p in zip(neutral_x, neutral_y):
                if p in self.targeted:
                    continue
                dist = numpy.linalg.norm(numpy.array(marine) - numpy.array(p))
                if not min_dist or dist < min_dist:
                    closest, min_dist = p, dist
            self.targeted.append(closest)
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
