"""
    Created by Michael McGuire 07/31/2019

    Edit this file to train an agent. You can control map, network, and training
    configuration.

"""


import sys
sys.path.insert(0, "../../interface/")

from custom_env import FullStateActionEnvironment
from Network import BaseNetwork
from base_agent import BaseAgent
from abstract_core import Experiment, RunSettings
from memory import ReplayMemory
from agent import AgentSettings
from sc2env_utils import env_config
import torch
import argparse


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():

    ### Change this map if you must
    map_name = "DefeatRoaches"
    render = False
    step_mul = 8


    ### Edit this to be a list of sc2_env.Agent() variables, one for each agent
    ### or bot you want, unless you are playing a minigame
    players = None


    env = FullStateActionEnvironment(map_name_=map_name,
                                render=render,
                                step_multiplier=step_mul,
                                players=players)


    ### Set this to construct your desired network inheriting from BaseNetwork
    model = None

    ### Change these parameters and dicts to customize training

    lr = 1e-4
    eps_max = 0.3
    eps_min = 0.05
    eps_duration=1e5
    history_size=20


    num_episodes = 1000000
    num_epochs = 2
    batch_size = 32
    train_every = 2048
    save_every = 10240
    graph_every = 50
    averaging_window = 100

    """
        :param optimizer: A class from torch.optim (instantiated later)
        :param learning_rate: The learning rate for the network
        :param epsilon_max: The starting epsilon
        :param epsilon_min: The final epsilon
        :param epsilon_duration: The number of frames to reach the final epsilon
    """
    agent_settings = AgentSettings(torch.optim.Adam,
                                lr,
                                eps_max,
                                eps_min,
                                eps_duration)

    ### Unless you are changing code in interface, you shouldn't change this dict
    run_settings = RunSettings(num_episodes,
                                num_epochs,
                                batch_size,
                                train_every,
                                save_every,
                                graph_every,
                                averaging_window)

    ### Unless you are changing memory, you shouldn't change this
    memory = ReplayMemory(train_every, batch_size, hist_size=history_size)
    """
    Custom to how you want to train your agent.
    Unless you are changing base_agent and changing the training algorithm,
    or you want to tune train parameters, you should not change this dict.
    """
    train_settings = {
        "discount_factor": 0.99,
        "lambda": 0.95,
        "hist_size": history_size,
        "device": device,
        "eps_denom": 1e-6,
        "c1": 0.1,
        "c2": 0.05,
        "c3": 0.01,
        "c4": 0.01,
        "clip_param": 0.1,
        "map": map_name
    }

    """
    Constructs the agent and trains it in an experiment.
    """
    agent = BaseAgent(model, agent_settings, memory, train_settings)
    experiment = Experiment([agent], env, run_settings)
    experiment.train()


if __name__ == "__main__":
    main()
