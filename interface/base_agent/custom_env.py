
from pysc2.env import sc2_env
from pysc2.lib import features, protocol
import numpy as np
from abstract_core import CustomEnvironment
from base_agent.process_state import state_processor
from base_agent.process_action import action_to_pysc2
from base_agent.sc2env_utils import env_config

from pysc2.lib.actions import FUNCTIONS
from base_agent.sc2env_utils import env_config

"""
    Generalized environment that uses a preprocessed version of the full state,
    and the full action space. Currently only supports single agent maps.
"""
class FullStateActionEnvironment(CustomEnvironment):

    def __init__(self, map_name_, render=False, step_multiplier=None, players=None):

        import sys
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)

        self.map = map_name_
        self.state_modifier_func = state_processor

        self._env = sc2_env.SC2Env(
            map_name=self.map,
            players=players,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=env_config["screen_width"],
                                                        minimap=env_config["minimap_width"]),
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
            self._curr_frame = self._env.step(
                [raw_action])[0]  # get obs for 1st agent
        except protocol.ConnectionError:
            self._curr_frame = self._env.reset()[0]


class RandomEnvironment(CustomEnvironment):
    """
    An environment that returns random states for testing purposes
    """

    def __init__(self, num_players=1, min_episode_len=1, max_episode_len=100,
            max_state_value=1):
        self.num_players = num_players
        self.min_episode_len = min_episode_len
        self.max_episode_len = max_episode_len
        self.max_state_value = max_state_value
        self.reset()

    def reset(self):
        self.steps_to_terminal = np.random.randint(self.min_episode_len,
            self.max_episode_len + 1)
        states = [self._gen_state() for i in range(self.num_players)]
        rewards = [self._gen_reward() for i in range(self.num_players)]
        return states, rewards, self.steps_to_terminal <= 0, None

    def step(self, _action):
        self.steps_to_terminal -= 1
        states = [self._gen_state() for i in range(self.num_players)]
        rewards = [self._gen_reward() for i in range(self.num_players)]
        return states, rewards, self.steps_to_terminal <= 0, None

    def _gen_reward(self):
        """
        80%: 0
        10%: 10
        10%: -1
        """
        rng = np.random.random()
        if rng < 0.8:
            return 0
        elif rng < 0.9:
            return 10
        else:
            return -1

    def _gen_state(self):
        """
        Generates random arrays with the same shapes as returned by
        FullStateActionEnvironment.
        """

        minimap_shape = (1, env_config['raw_minimap'],
            env_config['minimap_width'], env_config['minimap_width'])
        minimap = np.random.randint(self.max_state_value + 1, size=minimap_shape)

        screen_shape = (1, env_config['raw_screen'],
            env_config['screen_width'], env_config['screen_width'])
        screen = np.random.randint(self.max_state_value + 1, size=screen_shape)

        player_shape = (1, env_config['raw_player'])
        player = np.random.randint(self.max_state_value + 1, size=player_shape)

        # Mark all actions as available
        avail_actions = np.ones(len(FUNCTIONS))

        state = np.array([minimap, screen, player, avail_actions])
        return state
