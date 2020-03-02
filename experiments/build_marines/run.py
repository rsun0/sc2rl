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

def run_training():
    testing = True
    if testing:
        render = True
        verbose = True
        use_test_net = True
        use_test_agent = False
    else:
        render = False
        verbose = False
        use_test_net = False
        use_test_agent = False
    step_mul = 16

    lr = 0.0001
    discount = 1.0
    memsize = 320000
    num_blocks = 4
    num_channels = 32
    opt_eps = 1e-8

    num_episodes = 10
    num_epochs = 1
    batch_size = 32
    train_every = 1024
    save_every = train_every * 10
    graph_every = 0
    averaging_window = 100
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
        
        if use_test_agent:
            agent = TestAgent()
        else:
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
    run_training()
