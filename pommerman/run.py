import sys
sys.path.insert(0, "../interface/")

import copy

import pommerman.agents
import torch.optim

from abstract_core import Experiment, RunSettings
from agent import AgentSettings
from custom_env import PommermanEnvironment
from simple_agent import SimpleAgent
from noop_agent import NoopAgent, PommermanNoopAgent
from random_agent import RandomAgent
from mcts_agent import MCTSAgent, PolicyNetMemory
from pg_agent import PolicyGradientAgent
from policy_net import MCTSPolicyNet

if __name__ == '__main__':
    env = PommermanEnvironment(
        render=True,
        num_agents=2,
        game_state_file='start.json',
    )

    run_settings = RunSettings(
        num_episodes=10000,
        num_epochs=1,
        batch_size=32,
        train_every=512,
        save_every=2048,
        graph_every=50,
        averaging_window=50,
        graph_file='pommerman_results.png'
    )

    agent_settings = AgentSettings(
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        opt_eps=1e-8,
        epsilon_max=0,
        epsilon_min=0,
        epsilon_duration=0,
    )

    discount = 0.95
    memory = PolicyNetMemory(buffer_len=8192, discount=discount)

    mcts_model = MCTSPolicyNet(board_size=8, in_channels=20)
    agent1 = MCTSAgent(
        discount=discount,
        opponent=pommerman.agents.RandomAgent(),
        agent_id=0,
        tree_save_file='mct.pickle',
        model_save_file='policynet.h5',
        model=mcts_model,
        settings=agent_settings,
        memory=memory,
    )
    agent1.load()

    agent2 = RandomAgent()

    experiment = Experiment([agent1, agent2], env, run_settings)
    experiment.train()