from pysc2.env import sc2_env
from pysc2.lib import features, protocol
from action_interface import BuildMarinesAction, BuildMarinesActuator
from enum import Enum
from abstract_core import CustomEnvironment

class BuildMarinesEnvironment(CustomEnvironment):
    SCREEN_SIZE = 84
    MINIMAP_SIZE = 1
    MAP = 'BuildMarines'

    def __init__(self, render=False, step_multiplier=None):
        '''
        Initializes internal pysc2 environment
        :param render: Whether to render the game
        :param step_multiplier: Step multiplier for pysc2 environment
        '''

        import sys
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)

        self._env = sc2_env.SC2Env(
            map_name=self.MAP,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=self.SCREEN_SIZE, minimap=self.MINIMAP_SIZE),
                use_feature_units=True
            ),
            step_mul=step_multiplier,
            visualize=render,
            game_steps_per_episode=None
        )
        self._actuator = BuildMarinesActuator()
        self._prev_frame = None
        self._curr_frame = None
        self._terminal = True

    def reset(self):
        '''
        Resets the environment for a new episode
        :returns: Observations, reward, terminal, None for start state
        '''
        self._actuator.reset()
        self._terminal = False

        self._reset_env()
        self._terminal = self._curr_frame.last()
        agent_obs = self._curr_frame

        return [agent_obs], [self._curr_frame.reward], self._curr_frame.last(), [None]

    def step(self, action_list):
        '''
        Runs the environment until the next agent action is required
        :param action: 0 for Action.RETREAT or 1 for Action.ATTACK
        :returns: Observations, reward, terminal, None
        '''
        assert not self._terminal, 'Environment must be reset after init or terminal'
        
        action = action_list[0]
        # Convert to Enum
        action = BuildMarinesAction(action)

        self._run_to_next(action)
        self._terminal = self._curr_frame.last()
        agent_obs = self._curr_frame
        return [agent_obs], [self._curr_frame.reward], self._curr_frame.last(), [None]

    def _run_to_next(self, start_action):
        raw_action = self._actuator.compute_action(start_action, self._curr_frame)
        self._step_env(raw_action)
        
        while self._actuator.in_progress is not None:
            if self._curr_frame.last():
                return

            raw_action = self._actuator.compute_action(start_action, self._curr_frame)
            self._step_env(raw_action)
    
    def _reset_env(self):
        self._prev_frame = self._curr_frame
            # Get obs for 1st agent
        self._curr_frame = self._env.reset()[0]

    def _step_env(self, raw_action):
        self._prev_frame = self._curr_frame
        try:
            # Get obs for 1st agent
            self._curr_frame = self._env.step([raw_action])[0]
        except protocol.ConnectionError:
            self._curr_frame = self._env.reset()[0]
