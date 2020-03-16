import matplotlib
matplotlib.use('Agg')

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
    
    parser.add_argument('--log-file', type=str, default='bin/log.txt', help='log file')
    parser.add_argument('--graph-file', type=str, default='bin/graph.png', help='graph save location')
    parser.add_argument('--model-file', type=str, default='bin/model.h5', help='model save file')

    parser.add_argument('--lr', type=float, default=0.0001, help='starting learning rate')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='learning rate change')
    parser.add_argument('--lr-step-size', type=int, default=100, help='steps per lr change')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--init-temp', type=float, default=1.0)
    parser.add_argument('--temp-steps', type=int, default=16)
    parser.add_argument('--memsize', type=int, default=64000, help='experience replay memory size')
    parser.add_argument('--resblocks', type=int, default=4, help='number of resblocks in net')
    parser.add_argument('--channels', type=int, default=32, help='number of conv channels in net')
    
    parser.add_argument('--episodes', type=int, default=30000, help='number of episodes per epoch')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-every', type=int, default=16000, help='training period in number of steps')
    parser.add_argument('--save-every', type=int, default=160000, help='save period in number of steps')
    parser.add_argument('--graph-every', type=int, default=50, help='graphing period in number of episodes')
    parser.add_argument('--window', type=int, default=100, help='averaging window for graph')

    parser.add_argument('--render', action='store_true', help='render game')
    parser.add_argument('--verbose', action='store_true', help='enable printouts')
    parser.add_argument('--testagent', action='store_true', help='use test agent')
    parser.add_argument('--no-scvs', action='store_true', help='disable make SCV helper')
    parser.add_argument('--no-kill', action='store_true', help='disable kill marine helper')

    args = parser.parse_args()
    return args


def run_training(args):
    step_mul = 16
    opt_eps = 1e-8

    with open(args.log_file, mode='w') as log_file:
        # Removes "Namespace" from printout
        print('Args:', str(args)[9:], file=log_file)

        env = BuildMarinesEnvironment(
            render=args.render,
            step_multiplier=step_mul,
            verbose=args.verbose,
            enable_scv_helper=(not args.no_scvs),
            enable_kill_helper=(not args.no_kill),
        )
        run_settings = RunSettings(
            num_episodes=args.episodes,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            train_every=args.train_every,
            save_every=args.save_every,
            graph_every=args.graph_every,
            averaging_window=args.window,
            graph_file=args.graph_file,
            log_file=log_file,
            verbose=args.verbose,
        )
        
        if args.testagent:
            agent = TestAgent()
        else:
            agent_settings = AgentSettings(
                optimizer=torch.optim.Adam,
                learning_rate=args.lr,
                lr_gamma=args.lr_gamma,
                lr_step_size=args.lr_step_size,
                opt_eps=opt_eps,
                epsilon_max=0,
                epsilon_min=0,
                epsilon_duration=0,
                verbose=args.verbose,
            )
            memory = PolicyGradientMemory(
                buffer_len=args.memsize,
                discount=args.discount,
                averaging_window=args.window)
            model = PolicyGradientNet(
                num_blocks=args.resblocks,
                channels=args.channels,
            )
            agent = PolicyGradientAgent(
                init_temp=args.init_temp,
                temp_steps=args.temp_steps,
                save_file=args.model_file,
                log_file=log_file,
                model=model,
                settings=agent_settings,
                memory=memory,
            )
            agent.load()

        experiment = Experiment([agent], env, run_settings)
        experiment.train()


if __name__ == "__main__":
    args = parse_hyperparams()
    run_training(args)
