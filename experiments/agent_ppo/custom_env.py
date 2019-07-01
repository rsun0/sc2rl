from pysc2.env import sc2_env
from pysc2.lib import features, protocol
from action_interface import Action, Actuator
import numpy as np
from enum import Enum


class MinigameEnvironment:

    screen_width = 84
    action_width = 64

    def __init__(self, state_modifier_func, map_name_="DefeatRoaches", render=False, step_multiplier=None):
        '''
        Initializes internal pysc2 environment
        :param render: Whether to render the game
        :param step_multiplier: Step multiplier for pysc2 environment
        '''

        import sys
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)

        self.map = map_name_
        self.state_modifier_func = state_modifier_func

        self._env = sc2_env.SC2Env(
            map_name=map_name_,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=84),
                use_feature_units=True
            ),
            step_mul=step_multiplier,
            visualize=render,
            game_steps_per_episode=None
        )
        self._actuator = Actuator()
        self._prev_frame = None
        self._curr_frame = None
        self._terminal = True

        FACTOR = 9  # TODO
        self.observation_space = [84, 84, FACTOR]
        self.select_space = Actuator._SELECT_SPACE
        self.action_space = Actuator._ACTION_SPACE

    def reset(self):
        '''
        Resets the environment for a new episode
        :returns: Observations, reward, terminal, None for start state
        '''
        self._actuator.reset()
        self._terminal = False

        self._run_to_next()
        self._terminal = self._curr_frame.last()
        agent_obs = self._curr_frame
        #agent_obs = self._combine_frames()
        #agent_obs = self.state_modifier_func(self._curr_frame)

        size = len(agent_obs)
        """
        if (size == 4):
            info = agent_obs[-1]
            agent_obs = agent_obs[:-1]
        else:
            info = None
        """
        info = None
        return [agent_obs], [self._curr_frame.reward], [self._curr_frame.last()], [info]  # exclude selected

    def step(self, action):
        '''
        Runs the environment until the next agent action is required
        :param action: 0 for Action.RETREAT or 1 for Action.ATTACK
        :returns: Observations, reward, terminal, None
        '''
        topleft = action[0][0][0]
        botright = action[0][0][1]
        action = action[0][1]
        
        assert not self._terminal, 'Environment must be reset after init or terminal'
        assert action in range(5), 'Agent action must be 0-10'

        if action == 0:
            step_act = Action.SELECT.value
        elif action == 1:
            step_act = Action.ATTACK.value
        elif action == 2:
            step_act = Action.MOVE.value
        elif action == 3:
            step_act = Action.STOP.value
        elif action == 4:
            step_act = Action.NO_OP.value
        else:
            step_act = 0
        topleft = self.convert_spatial(topleft)
        botright = self.convert_spatial(botright)
        self._run_to_next(step_act, topleft=topleft, botright=botright)
        self._terminal = self._curr_frame.last()
        #agent_obs = self._combine_frames()
        #agent_obs = self.state_modifier_func(self._curr_frame)
        agent_obs = self._curr_frame
        """
        if (len(agent_obs) == 4):
            info = agent_obs[-1]
            agent_obs = agent_obs[:-1]
        else:
            info = None
        """
        info = None
        return [agent_obs], [self._curr_frame.reward], [self._curr_frame.last()], [info]  # exclude selected

    def _run_to_next(self, start_action=None, topleft=None, botright=None):

        if start_action is None:
            self._reset_env()
            start_action = Action.SELECT.value

        if self._curr_frame.last():
            return

        raw_action = self._actuator.compute_action(
            start_action, self._curr_frame, topleft=topleft, botright=botright)
        self._step_env(raw_action)

        """
        # Select action
        if (topleft is not None):
            #print("Selecting")
            friendly_unit_density = custom_obs[2]
            assert not np.all(friendly_unit_density == 0), 'All marines dead but not terminal state'
            selected = custom_obs[0]
            #while not self._actuator.units_selected or np.all(selected == 0):
            
            raw_action = self._actuator.compute_action(Action.SELECT, custom_obs, self._curr_frame, topleft=topleft, botright=botright)
            self._step_env(raw_action)
            if self._curr_frame.last():
                return
            custom_obs = self.state_modifier_func(self._curr_frame)
            selected = custom_obs[0]
                
            #assert self._actuator.units_selected and np.any(selected > 0), 'Units not selected after select action'
        """

        """
        # Move action
        if (start_action is not None):
            #print("Moving")
            last_obs = self.state_modifier_func(self._curr_frame)
            raw_action = self._actuator.compute_action(start_action, last_obs, self._curr_frame, topleft=topleft, botright=botright)
            self._step_env(raw_action)
        """

    def _combine_frames(self):
        '''
        Combines the previous and current frame for observations
        '''
        assert self._prev_frame is not None and self._curr_frame is not None, 'Returning to agent after less than 2 frames should be impossible'

        custom_prev = self.state_modifier_func(self._prev_frame)[1:]
        custom_curr = self.state_modifier_func(self._curr_frame)
        # move selected frame to end
        custom_curr = custom_curr[np.r_[1:len(custom_curr), 0]]
        custom_frames = np.append(custom_prev, custom_curr, axis=0)
        return custom_frames

    def _reset_env(self):
        self._prev_frame = self._curr_frame
        self._curr_frame = self._env.reset()[0]  # get obs for 1st agent

    def _step_env(self, raw_action):
        self._prev_frame = self._curr_frame
        try:
            self._curr_frame = self._env.step(
                [raw_action])[0]  # get obs for 1st agent
        except protocol.ConnectionError:
            self._curr_frame = self._env.reset()[0]
            
            
    def convert_spatial(self, coords):
        if coords is None:
            return None
        new_coords = []
        for i in coords:
            new_coords.append((i / MinigameEnvironment.action_width) * MinigameEnvironment.screen_width)
        return new_coords
