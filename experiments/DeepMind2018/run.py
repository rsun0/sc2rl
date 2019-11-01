import sys
sys.path.insert(0, "../../interface/")

from base_agent.custom_env import FullStateActionEnvironment
from RRLNetwork import RRLModel
from base_agent.base_agent import BaseAgent
from abstract_core import Experiment, RunSettings
from base_agent.memory import ReplayMemory, SequentialMemory
from agent import AgentSettings
from base_agent.sc2env_utils import env_config, full_action_space
import torch
import argparse
import numpy as np


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmarks = True

def main():

    map_name = "DefeatRoaches"
    render = False
    step_mul = 8


    env = FullStateActionEnvironment(map_name_=map_name,
                                render=render,
                                step_multiplier=step_mul)

    state_embed = 10
    action_embed = 16

    lr = 1e-4
    opt_eps = 1e-8
    eps_max = 0.3
    eps_min = 0.05
    eps_duration=2e4
    history_size=5

    num_episodes = 10000000
    num_epochs = 3
    batch_size = 32
    train_every = 1024
    save_every = 10240
    graph_every = 50
    averaging_window = 100

    net_config = {
        "state_embedding_size": state_embed, # number of features output by embeddings
        "action_embedding_size": action_embed,
        "down_conv_features": 128,
        "down_residual_depth": 2,
        "up_features": 32,
        "up_conv_features": 64,
        "resnet_features": 128,
        "LSTM_in_size": 64,
        "LSTM_hidden_size": 96,
        "inputs2d_size": 64,
        "inputs3d_width": 8,
        "relational_features": 32,
        "relational_depth": 5,
        "relational_heads": 3,
        "spatial_out_depth": 64,
        "channels3": 16,
        "history_size": history_size,
        "device": device
    }

    #action_space = np.zeros(full_action_space.shape)
    #action_space[[0, 3, 12, 13, 331, 332]] = 1
    action_space = np.ones(full_action_space.shape)
    model = RRLModel(net_config, device=device, action_space=action_space).to(device)
    print(model)

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
                                eps_duration,
                                opt_eps)

    run_settings = RunSettings(num_episodes,
                                num_epochs,
                                batch_size,
                                train_every,
                                save_every,
                                graph_every,
                                averaging_window)

    memory = ReplayMemory(train_every, batch_size, hist_size=history_size)

    train_settings = {
        "discount_factor": 0.99,
        "lambda": 0.95,
        "hist_size": history_size,
        "device": device,
        "eps_denom": 1e-5,
        "c1": 0.1,
        "c2": 0.1,
        "c3": 0.3,
        "c4": 0.3,
        "minc2": 0.01,
        "clip_param": 0.1,
        "min_clip_param": 0.01,
        "clip_decay": 10000,
        "c2_decay": 10000,
        "map": map_name,
        "history_size": history_size
    }

    agent = BaseAgent(model, agent_settings, memory, train_settings)
    #agent.load()
    experiment = Experiment([agent], env, run_settings)

    #experiment.test()
    experiment.train()


if __name__ == "__main__":
    main()
