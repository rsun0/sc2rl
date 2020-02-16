import sys
sys.path.insert(0, "../../interface/")

from custom_env import BuildMarinesEnvironment
from abstract_core import Experiment, RunSettings
from agent import AgentSettings
from test_agent import TestAgent

import torch
import argparse

def run_training():
    # FIXME SCVs are not rallied at start on reset
    render = False
    step_mul = 16

    env = BuildMarinesEnvironment(render=render, step_multiplier=step_mul)
                                
    # lr = 5e-3
    # eps_max = 0.3
    # eps_min = 0.05
    # eps_duration=1e5
    
    # agent_settings = AgentSettings(torch.optim.Adam,
    #                             lr,
    #                             eps_max,
    #                             eps_min,
    #                             eps_duration)
    
    num_episodes = 1
    num_epochs = 1
    batch_size = 32
    train_every = 9999
    save_every = 9999
    graph_every = 0
    averaging_window = 100
    
    run_settings = RunSettings(num_episodes,
                                num_epochs,
                                batch_size,
                                train_every,
                                save_every,
                                graph_every,
                                averaging_window)
    
    agent = TestAgent()
    experiment = Experiment([agent], env, run_settings)
    
    experiment.train()


if __name__ == "__main__":
    run_training()
