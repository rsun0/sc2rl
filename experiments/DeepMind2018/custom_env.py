
from pysc2.env import sc2_env
from pysc2.lib import features, protocol
import numpy as np
from abstract_core import CustomEnvironment
from process_state import state_processor
from process_action import action_to_pysc2

"""
    Generalized environment that uses a preprocessed version of the full state,
    and the full action space. Currently only supports single agent maps.
"""
class FullStateActionEnvironment(CustomEnvironment):

    def __init__(self, map_name_, render=False, step_multiplier=None):

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
            ),
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
                [self._curr_frame.reward], self._curr_frame.last(), [info]

    def step(self, action):

        assert (not self._terminal)
        self._run_to_next(action)
        agent_obs = self.state_modifier_func(self._curr_frame)
        reward = self._curr_frame.reward
        done = self._curr_frame.last()
        info = None
        return [agent_obs], [reward], done, [info]

    def _run_to_next(self, action=None, reset=False):
        if reset:
            self._reset_env()
            return

        if self._curr_frame.last():
            return

        raw_action = self.action_modifier_func(action)
        self._step_env(raw_action)

    def _reset_env(self):
        self._curr_frame = self._env.reset()[0]

    def _step_env(self, raw_action):
        self._prev_frame = self._curr_frame
        try:
            print(raw_action)
            self._curr_frame = self._env.step(
                [raw_action])[0]  # get obs for 1st agent
        except protocol.ConnectionError:
            self._curr_frame = self._env.reset()[0]
