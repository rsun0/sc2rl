import sys
sys.path.insert(0, "../interface/")

import copy

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from agent import Model
from mcts_agent import MCTSAgent


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.main = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        identity = x
        out = self.main(x)
        out += identity
        out = torch.relu_(out)
        return out


class ActorCriticNet(nn.Module, Model):
    def __init__(self,
            board_size,
            in_channels,
            num_blocks=4,
            channels=32,
            num_actions=6):
        super().__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        self.num_actions = num_actions

        # Convolutions
        convs = [
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ]
        for i in range(num_blocks):
            convs.append(ResBlock(channels, channels))
        convs.extend([
            nn.Conv2d(channels, 4, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        ])
        self.shared_convs = nn.Sequential(*convs)
        
        fc_h = 4 * board_size ** 2
        self.shared_fcs = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, 1),
            nn.Tanh(),
        )

        self.actor_criterion = nn.CrossEntropyLoss()
        self.critic_criterion = nn.MSELoss()

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).type(torch.FloatTensor)
        x = self.shared_convs(state)
        x = torch.flatten(x, start_dim=1)
        x = self.shared_fcs(x)
        policy_scores = self.actor(x)
        vals = self.critic(x)
        return policy_scores, vals

    def get_batched_greedy_actions(self, batched_env_states, env):
        self.eval()
        agent_id = env.training_agent
        batched_greedy_actions = []
        for env_states in batched_env_states:
            greedy_actions = np.empty(len(env_states), dtype=int)
            for i, env_state in enumerate(env_states):
                next_states = np.empty((self.num_actions, self.in_channels, self.board_size, self.board_size))

                MCTSAgent.set_state(env, env_state)
                obs = env.get_observations()
                actions = env.act(obs)
                actions.insert(agent_id, None)

                terminal = np.full(self.num_actions, False)
                terminal_rewards = np.empty(self.num_actions)
                for a in range(self.num_actions):
                    actions[env.training_agent] = a
                    obs, rewards, done, _ = env.step(actions)
                    if done:
                        terminal[a] = True
                        terminal_rewards[a] = rewards[agent_id]
                    state, _ = MCTSAgent.state_space_converter(obs[agent_id])
                    next_states[a] = state

                    MCTSAgent.set_state(env, env_state)

                if terminal.all():
                    vals = terminal_rewards
                else:
                    _, vals = self(next_states)
                    vals = vals.detach().numpy()[:, 0]
                    # Replace vals with reward if terminal
                    vals[terminal] = terminal_rewards[terminal]
                greedy_actions[i] = np.argmax(vals)
            batched_greedy_actions.append(greedy_actions)
        return batched_greedy_actions

    def optimize(self, data, batch_size, optimizer, env):
        # TODO include metrics
        if len(data) < 2:
            return

        self.train()
        dtype = next(self.parameters()).type()
        # Preserve existing state of env
        saved_state = MCTSAgent.get_state(env)
        
        batched_env_states = []
        batched_states = []
        pbar = tqdm(range(0, len(data), batch_size))
        for i in pbar:
            batch = data[i:i + batch_size]
            if len(batch) == 1:
                # Batch norm will fail
                break
            
            raw_states, _, raw_true_vals, env_states = zip(*batch)
            states = torch.from_numpy(np.stack(raw_states)).type(dtype)
            true_vals = torch.from_numpy(np.array(raw_true_vals)[:, np.newaxis]).type(dtype)

            optimizer.zero_grad()
            _, vals = self(states)
            loss = self.critic_criterion(vals, true_vals)
            loss.backward()
            optimizer.step()

            batched_env_states.append(env_states)
            batched_states.append(states)

        batched_greedy_actions = self.get_batched_greedy_actions(batched_env_states, env)
        # get_batched_greedy_actions calls eval()
        self.train()
        for states, greedy_actions in zip(batched_states, batched_greedy_actions):
            greedy_actions = torch.from_numpy(greedy_actions)

            optimizer.zero_grad()
            pi_scores, _ = self(states)
            loss = self.actor_criterion(pi_scores, greedy_actions)
            loss.backward()
            optimizer.step()

        # Restore existing state before training
        MCTSAgent.set_state(env, saved_state)