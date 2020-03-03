import argparse

import sys
sys.path.insert(0, "../../interface/")

from custom_env import BuildMarinesEnvironment
from abstract_core import Experiment, RunSettings
from agent import AgentSettings
from pg_agent import PolicyGradientAgent, PolicyGradientMemory
from test_agent import TestAgent
from policy_net import PolicyGradientNet

import torch
import argparse


def parse_hyperparams():
    parser = argparse.ArgumentParser()

    parser.add_argument('--render', action='store_true', default=False, help='render game')
    parser.add_argument('--verbose', action='store_true', default=False, help='enable printouts')
    parser.add_argument('--testnet', action='store_true', default=False, help='use simple net')
    parser.add_argument('--testagent', action='store_true', default=False, help='use test agent')

    args = parser.parse_args()
    return args


def run_training(args):
    render = args.render
    verbose = args.verbose
    use_test_net = args.testnet
    use_test_agent = args.testagent
    step_mul = 16

    lr = 0.0001
    discount = 1.0
    memsize = 320000
    num_blocks = 4
    num_channels = 32
    opt_eps = 1e-8

    num_episodes = 10000
    num_epochs = 1
    batch_size = 32
    train_every = 1024
    save_every = train_every * 10
    graph_every = 10
    averaging_window = 20
    graph_file = 'bin/graph.png'
    save_file = 'bin/model.h5'
    log_filename = 'bin/log.txt'

    with open(log_filename, mode='w') as log_file:
        env = BuildMarinesEnvironment(
            render=render,
            step_multiplier=step_mul,
            verbose=verbose,
            enable_scv_helper=True,
            enable_kill_helper=True,
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
            log_file=log_file,
            verbose=verbose,
        )
        
        if use_test_agent:
            agent = TestAgent()
        else:
            agent_settings = AgentSettings(
                optimizer=torch.optim.Adam,
                learning_rate=lr,
                opt_eps=opt_eps,
                epsilon_max=0,
                epsilon_min=0,
                epsilon_duration=0,
                verbose=verbose,
            )
            memory = PolicyGradientMemory(
                buffer_len=memsize,
                discount=discount,
                averaging_window=averaging_window)
            model = PolicyGradientNet(
                num_blocks=num_blocks,
                channels=num_channels,
                test_mode=use_test_net,
            )
            agent = PolicyGradientAgent(
                save_file=save_file,
                model=model,
                settings=agent_settings,
                memory=memory,
            )
            agent.load()

        experiment = Experiment([agent], env, run_settings)
        experiment.train()


if __name__ == "__main__":
    args = parse_hyperparams()
    print('Args: ', args)
    run_training(args)
