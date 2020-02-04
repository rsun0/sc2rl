import sys
sys.path.insert(0, "../interface/")

from abstract_core import CustomEnvironment
import pommerman
from pommerman import agents


class DummyAgent(agents.BaseAgent):
    """
    Placeholder for learning agent
    """
    def act(self, obs, action_space):
        return 0


class PommermanEnvironment(CustomEnvironment):

    def __init__(self, render=False, num_agents=1, game_state_file=None):
        self.render = render

        self.num_agents = num_agents
        if self.num_agents == 1:
            agent_list = [
                DummyAgent(),
                agents.SimpleAgent(),
            ]
        elif self.num_agents == 2:
            agent_list = [
                DummyAgent(),
                DummyAgent(),
            ]
        self._env = pommerman.make('OneVsOne-v0', agent_list)
        self._env.set_init_game_state(game_state_file)
        # For saving initial board
        # self._env.reset()
        # self._env.save_json('.')
        self._state = None

    def reset(self):
        self._state = self._env.reset()
        obs = self._state[:self.num_agents]
        rewards = [0 for i in range(self.num_agents)]
        if self.render:
            self._env.render()
        return obs, rewards, False, None

    def step(self, action_list):
        if self.num_agents == 1:
            actions = self._env.act(self._state)
            actions[:1] = action_list
        else:
            actions = action_list
        self._state, reward, terminal, info = self._env.step(actions)
        
        if self.render:
            self._env.render()
        
        return self._state[:self.num_agents], reward[:self.num_agents], terminal, info