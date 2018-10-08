from pysc2.env import sc2_env
from pysc2.lib import features
from modified_state_space import state_modifier
from action_interface import Action, Actuator
from numpy import all as np_all
import numpy as np

class DefeatRoachesEnvironment:

    def __init__(self, render=False, step_multiplier=1):
        '''
        Initializes internal pysc2 environment
        :param render: Whether to render the game
        :param step_multiplier: Step multiplier for pysc2 environment
        '''
        self.env = sc2_env.SC2Env(
            map_name='DefeatRoaches',
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True
            ),
            step_mul=step_multiplier,
            visualize=render,
            game_steps_per_episode=0
        )
        self.actuator = Actuator()
        self.last_obs = None
        self.action_space = 2
        FACTOR = 8 # TODO
        self.observation_space = 84*84*FACTOR # 

    def reset(self):
        '''
        Resets the environment for a new episode
        :returns: The start state
        '''
        self.actuator.reset()
        raw_obs = self._run_to_next()
        assert not raw_obs.last(), 'Environment reset immediately led to terminal state'
        custom_obs, _ = state_modifier.modified_state_space(raw_obs)
        self.last_obs = custom_obs
        return custom_obs[1:] # exclude selected

    def step(self, action):
        '''
        Runs the environment until the next agent action is required
        :param action: Action.ATTACK or Action.RETREAT
        :returns: Observations, reward, terminal, None
        '''
        
        assert self.last_obs is not None, 'Environment must be reset after init or terminal'
        #assert action == Action.ATTACK or action == Action.RETREAT, 'Agent action must be attack or retreat'
        if (action == 0):
            step_act = Action.RETREAT
        if (action == 1):
            step_act = Action.ATTACK
        
        
        raw_obs = self._run_to_next(step_act)
        custom_obs, _ = state_modifier.modified_state_space(raw_obs)
        self.last_obs = custom_obs if not raw_obs.last() else None
        return custom_obs[1:], raw_obs.reward, raw_obs.last(), None

    def _run_to_next(self, start_action=None):
        '''
        Runs the environment with NO_OPs and SELECTs until the next agent action is required
        :param start_action: The chosen agent action, or None for reset
        :returns: Final raw observations
        '''
        if start_action is None:
            raw_obs = self.env.reset()[0] # get obs for 1st agent
        else:
            raw_action = self.actuator.compute_action(start_action, self.last_obs)
            raw_obs = self.env.step([raw_action])
        
        if raw_obs.last():
            return raw_obs
        
        custom_obs, _ = state_modifier.modified_state_space(raw_obs)

        friendly_unit_density = custom_obs[2]
        assert not np_all(friendly_unit_density == 0), 'All marines dead but not terminal state'

        selected = custom_obs[0]
        if not self.actuator.units_selected or np_all(selected == 0):
            raw_action = self.actuator.compute_action(Action.SELECT, custom_obs)
            raw_obs = self.env.step([raw_action])
        assert self.actuator.units_selected, 'Units not selected after select action'
        
        return raw_obs
