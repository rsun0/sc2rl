import sys
sys.path.insert(0, "../interface/")

import pommerman.agents

from abstract_core import Experiment, RunSettings
from custom_env import PommermanEnvironment
from simple_agent import SimpleAgent
from noop_agent import NoopAgent
from random_agent import RandomAgent
from mcts_agent import MCTSAgent

if __name__ == '__main__':
    env = PommermanEnvironment(render=True, num_agents=2, game_state_file='start.json')

    run_settings = RunSettings(
        num_episodes=10000,
        num_epochs=1,
        batch_size=1,
        train_every=1024,
        save_every=64,
        graph_every=1,
        averaging_window=10,
        graph_file='pommerman_results.png'
    )

    agent1 = RandomAgent()
    agent2 = MCTSAgent(opponent=pommerman.agents.RandomAgent())
    agent2.load()

    experiment = Experiment([agent1, agent2], env, run_settings)
    experiment.train()