import sys
sys.path.insert(0, "../interface/")

import copy
import argparse

import pommerman.agents
import torch.optim

from abstract_core import Experiment, RunSettings
from agent import AgentSettings
from custom_env import PommermanEnvironment
from simple_agent import SimpleAgent
from noop_agent import NoopAgent, PommermanNoopAgent
from random_agent import RandomAgent
from mcts_agent import MCTSAgent, MCTSMemory
from pg_agent import PolicyGradientAgent
from policy_net import MCTSPolicyNet
from actorcritic_net import ActorCriticNet


def parse_hyperparams():
    parser = argparse.ArgumentParser()

    parser.add_argument('--opponent', type=str, choices=['rand', 'noop', 'simp'], default='rand', help='opponent type')
    
    parser.add_argument('--board-size', type=int, default=8, help='side length of game board')
    parser.add_argument('--board-file', type=str, default='start.json', help='starting state file')
    parser.add_argument('--graph-file', type=str, default='bin/graph.png', help='graph save location')
    parser.add_argument('--model-file', type=str, default='bin/model.h5', help='model save file')

    parser.add_argument('--searches', type=int, default=32, help='MCTS searches per turn')
    parser.add_argument('--temp', type=float, default=1.0, help='MCTS temperature')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--memsize', type=int, default=32000, help='experience replay memory size')

    parser.add_argument('--episodes', type=int, default=10000, help='number of episodes per epoch')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-every', type=int, default=256, help='training period in number of steps')
    parser.add_argument('--save-every', type=int, default=2048, help='save period in number of steps')
    parser.add_argument('--graph-every', type=int, default=100, help='graphing period in number of episodes')
    parser.add_argument('--window', type=int, default=200, help='averaging window for graph')

    parser.add_argument('--render', action='store_true', default=False, help='render game')
    parser.add_argument('--verbose', action='store_true', default=False, help='enable printouts')

    args = parser.parse_args()
    return args


def run_training(
        opponent,
        game_state_file,
        graph_file,
        model_save_file,
        mcts_iters,
        temp,
        lr,
        discount,
        memsize,
        num_episodes,
        num_epochs,
        batch_size,
        train_every,
        save_every,
        graph_every,
        averaging_window,
        opt_eps=1e-8,
        ucb_c=1.5,
        boardsize=8,
        inputs=20,
        render=False,
        verbose=False,
    ):
    env = PommermanEnvironment(
        render=render,
        num_agents=2,
        game_state_file=game_state_file,
    )

    run_settings = RunSettings(
        num_episodes=num_episodes,
        num_epochs=num_epochs,
        batch_size=batch_size,
        train_every=train_every,
        save_every=save_every,
        graph_every=graph_every,
        averaging_window=averaging_window,
        graph_file=graph_file,
        verbose=verbose,
    )

    agent_settings = AgentSettings(
        optimizer=torch.optim.Adam,
        learning_rate=lr,
        opt_eps=opt_eps,
        epsilon_max=0,
        epsilon_min=0,
        epsilon_duration=0,
        verbose=verbose,
    )

    memory = MCTSMemory(buffer_len=memsize, discount=discount)

    if opponent == 'rand':
        opp = pommerman.agents.RandomAgent()
        agent2 = RandomAgent()
    elif opponent == 'noop':
        opp = PommermanNoopAgent()
        agent2 = NoopAgent()
    elif opponent == 'simp':
        opp = pommerman.agents.SimpleAgent()
        agent2 = SimpleAgent()
    else:
        raise Exception('Invalid opponent type', opponent)

    mcts_model = ActorCriticNet(board_size=boardsize, in_channels=inputs)
    agent1 = MCTSAgent(
        mcts_iters=mcts_iters,
        discount=discount,
        c=ucb_c,
        temp=temp,
        agent_id=0,
        opponent=opp,
        model_save_file=model_save_file,
        model=mcts_model,
        settings=agent_settings,
        memory=memory,
    )
    agent1.load()

    experiment = Experiment([agent1, agent2], env, run_settings)
    experiment.train()


if __name__ == '__main__':
    args = parse_hyperparams()
    run_training(
        opponent=args.opponent,
        game_state_file=args.board_file,
        graph_file=args.graph_file,
        model_save_file=args.model_file,
        mcts_iters=args.searches,
        temp=args.temp,
        lr=args.lr,
        discount=args.discount,
        memsize=args.memsize,
        num_episodes=args.episodes,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        train_every=args.train_every,
        save_every=args.save_every,
        graph_every=args.graph_every,
        averaging_window=args.window,
        render=args.render,
        verbose=args.verbose,
        boardsize=args.board_size,
    )