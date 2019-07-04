import sys
sys.path.insert(0, "../../interface/")

from custom_env import MinigameEnvironment
from graphconv import GraphConvModel
from modified_state_space import state_modifier
from agent_ppo import PPOAgent
from abstract_core import Experiment, RunSettings
from memory import ReplayMemory
from agent import AgentSettings

import torch
import argparse



device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():

    map_name = "DefeatRoaches"
    render = True
    step_mul = 8


    env = MinigameEnvironment(state_modifier.graph_conv_modifier,
                                map_name_=map_name,
                                render=render,
                                step_multiplier=step_mul)
                                
    nonspatial_act_size, spatial_act_depth = env.action_space
                                
    model = GraphConvModel(nonspatial_act_size, spatial_act_depth, device=device).to(device)
    
    
    lr = 5e-3
    eps_max = 0.3
    eps_min = 0.05
    eps_duration=1e5
    
    
    num_episodes = 1000000
    num_epochs = 3
    batch_size = 32
    train_every = 1024
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
                                
    memory = ReplayMemory(train_every, 1, batch_size)
                                
    PPO_settings = {
        "discount_factor": 0.99,
        "lambda": 0.95,
        "hist_size": 8,
        "device": device,
        "eps_denom": 1e-6,
        "c1": 1.0,
        "c2": 0.5,
        "c3": 0.5,
        "c4": 1.0,
        "clip_param": 0.1
    }
    
    
    
    agent = PPOAgent(model, agent_settings, memory, PPO_settings)
    experiment = Experiment([agent], env, run_settings)
    
    experiment.train()


if __name__ == "__main__":
    main()
    


















