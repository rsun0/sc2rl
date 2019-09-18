import sys
sys.path.insert(0, "../interface/")

from abstract_core import CustomEnvironment
import pommerman
from pommerman import agents


class DummyAgent(agents.BaseAgent):
    def act(self, obs, action_space):
        return 0


class PommermanEnvironment(CustomEnvironment):

    def __init__(self, render=False):
        self.render = render

        agent_list = [
            DummyAgent(), # placeholder for learning agent
            agents.SimpleAgent(),
        ]
        self._env = pommerman.make('OneVsOne-v0', agent_list)
        self._state = None

    def reset(self):
        self._state = self._env.reset()
        return self._state[:1], [0], False, [None] 

    def step(self, action):
        if self.render:
            self._env.render()
        actions = self._env.act(self._state)
        actions[0] = action
        self._state, reward, terminal, info = self._env.step(actions)
        return self._state[:1], reward[:1], terminal, info