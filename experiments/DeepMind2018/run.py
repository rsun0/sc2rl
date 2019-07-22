import sys
sys.path.insert(0, "../../interface/")

from custom_env import FullStateActionEnvironment
from Network import RRLModel
from modified_state_space import state_modifier
from agent_ppo import PPOAgent
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


    env = MinigameEnvironment(map_name_=map_name,
                                render=render,
                                step_multiplier=step_mul)

    state_embed = 10
    action_embed = 16

    net_config = {
        "state_embedding_size": state_embed, # number of features output by embeddings
        "action_embedding_size": action_embed,
        "down_conv_features": 64,
        "up_features": 64,
        "up_conv_features": 256
        "resnet_features": 256,
        "LSTM_in_size": 128,
        "LSTM_hidden_size:" 256,
        "inputs2d_size": 128,
        "relational_features": 64
        "relational_depth": 3
    }

    model = RRLModel(net_config, device=device).to(device)
