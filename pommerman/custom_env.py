from interface.abstract_core import CustomEnvironment
import pommerman
from pommerman import agents


class DummyAgent(agents.BaseAgent):
    def act(self, obs, action_space):
        return 0


class PommermanEnvironment(CustomEnvironment):

    def __init__(self, render=False):
        self.render = render

        agent_list = [
            agents.BaseAgent(),
            agents.SimpleAgent(),
        ]
        self._env = pommerman.make('OneVsOne-v0', agent_list)

    def reset(self):
        self.state = self._env.reset()
        # TODO extract learning agent's state from state
        return [self.state], [0], False, [None] 

    def step(self, action):
        if self.render:
            self._env.render()
            actions = env.act(state)
            self._env.step(actions)