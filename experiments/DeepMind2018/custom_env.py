
from pysc2.env import sc2_env
from pysc2.lib import features, protocol
import numpy as np
from abstract_core import CustomEnvironment
from process_state import state_processor
from process_action import action_to_pysc2


class FullStateActionEnvironment(CustomEnvironment):

    def __init__(self, state_modifier_func, map_name_, render=False, step_multiplier=None):

        import sys
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)

        self.map = map_name_
        self.state_modifier_func = state_processor

        self._env = sc2_env.SC2Env(
            map_name=self.map,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=84),
                use_feature_units=True
            )
            step_mul=step_multiplier,
            visualize=render,
            game_steps_per_episode=None
        )
        self.action_modifier_func = action_to_pysc2
        self._curr_frame = None
        self._terminal = True

    def reset(self):
        self._terminal = False
        self._run_to_next(reset=True)
        self._terminal = self._curr_frame.last()
        agent_obs = self._curr_frame
        info = None
        return [self.state_modifier_func(self._curr_frame)],  \
                [self.curr_frame.reward], self.curr_frame.last(), [info]

    def step(self, action):
        self._run_to_next(action)
        agent_obs = self.state_modifier_func(self._curr_frame)
        reward = self._curr_frame.reward
        done = self._curr_frame.last()
        info = None
        return [agent_obs], [reward], done, [info]
