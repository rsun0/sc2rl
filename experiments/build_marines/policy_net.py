import sys
sys.path.insert(0, "../interface/")

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from agent import Model
from pg_agent import NUM_ACTIONS, NUM_CHANNELS
from custom_env import BuildMarinesEnvironment


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


class PolicyGradientNet(nn.Module, Model):
    def __init__(self,
            num_blocks=4,
            channels=32,
            screen_size=BuildMarinesEnvironment.SCREEN_SIZE,
            in_channels=NUM_CHANNELS,
            num_actions=NUM_ACTIONS):
        super().__init__()
        self.screen_size = screen_size
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
        self.convs = nn.Sequential(*convs)
        
        fc_h = 4 * screen_size ** 2
        self.fc = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, num_actions)
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).type(torch.FloatTensor)
        x = self.convs(state)
        x = torch.flatten(x, start_dim=1)
        policy_scores = self.fc(x)
        return policy_scores

    def optimize(self, data, batch_size, optimizer, verbose=False):
        if len(data) < 2:
            # Need at least 2 data points for batch norm
            return None
        
        self.train()
        running_loss = 0
        pbar = tqdm(range(0, len(data) - batch_size + 1, batch_size))
        for i in pbar:
            batch = data[i:i+batch_size]
            states, actions, rewards = zip(*batch)

            states_batch = np.stack(states)
            actions_batch = np.array(actions)
            rewards_batch = torch.from_numpy(np.array(rewards))

            actions_onehot = np.zeros((actions_batch.shape[0], NUM_ACTIONS))
            actions_onehot[np.arange(actions_batch.shape[0]), actions_batch] = 1
            actions_onehot = torch.from_numpy(actions_onehot)

            preds = self.model(states_batch)
            log_probs = torch.nn.functional.log_softmax(preds, dim=1)
            log_probs_observed = torch.sum(log_probs * actions_onehot, dim=1)
            loss = -torch.sum(log_probs_observed * rewards_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_experiences = i + batch_size
            pbar.set_postfix_str("{:.3f}L".format(running_loss / num_experiences))
        pbar.close()

        num_batches = np.ceil(len(data) / batch_size)
        avg_loss = running_loss / num_batches
        return avg_loss