import sys
sys.path.insert(0, "../../interface/")

from custom_env import FullStateActionEnvironment
from Network import RRLModel
from agent_rrl import RRLAgent
from abstract_core import Experiment, RunSettings
from memory import ReplayMemory
from agent import AgentSettings
from sc2env_utils import env_config
import torch
import argparse


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():

    map_name = "DefeatRoaches"
    render = False
    step_mul = 8


    env = FullStateActionEnvironment(map_name_=map_name,
                                render=render,
                                step_multiplier=step_mul)

    state_embed = 10
    action_embed = 16

    net_config = {
        "state_embedding_size": state_embed, # number of features output by embeddings
        "action_embedding_size": action_embed,
        "down_conv_features": 32,
        "up_features": 32,
        "up_conv_features": 128,
        "resnet_features": 128,
        "LSTM_in_size": 64,
        "LSTM_hidden_size": 96,
        "inputs2d_size": 64,
        "inputs3d_width": 8,
        "relational_features": 32,
        "relational_depth": 3,
        "relational_heads": 3,
        "spatial_out_depth": 64,
        "channels3": 16,
        "device": device
    }

    model = RRLModel(net_config, device=device).to(device)
    print(model)


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
        "hist_size": 12,
        "device": device,
        "eps_denom": 1e-6,
        "c1": 0.1,
        "c2": 0.05,
        "c3": 0.01,
        "c4": 0.01,
        "clip_param": 0.1,
        "map": map_name
    }

    agent = RRLAgent(model, agent_settings, memory, train_settings)
    experiment = Experiment([agent], env, run_settings)

    experiment.train()


if __name__ == "__main__":
    main()
