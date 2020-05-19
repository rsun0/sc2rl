import sys
sys.path.insert(0, "../interface/")

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from agent import Model
from pg_agent import NUM_IMAGES, NUM_SCALARS
from custom_env import SCREEN_SIZE
from action_interface import NUM_ACTIONS


class ResBlock(nn.Module):
    def __init__(self, innodes, nodes):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(innodes, nodes),
            nn.BatchNorm1d(nodes),
            nn.ReLU(inplace=True),
            nn.Linear(nodes, nodes),
            nn.BatchNorm1d(nodes),
        )

    def forward(self, x):
        identity = x
        out = self.main(x)
        out += identity
        out = torch.relu_(out)
        return out


class PolicyGradientNet(nn.Module, Model):
    def __init__(self,
            num_blocks=2,
            channels=32,
            force_cpu=False):
        super().__init__()

        self.force_cpu = force_cpu

        self.convs = nn.Sequential(
            nn.Conv2d(NUM_IMAGES, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        flattened_width = 84 * 84
        self.image_linears = nn.Sequential(
            nn.Linear(flattened_width, flattened_width // 24),
            nn.BatchNorm1d(flattened_width // 24),
            nn.ReLU(),
            nn.Linear(flattened_width // 24, NUM_IMAGES * 4),
            nn.BatchNorm1d(NUM_IMAGES * 4),
            nn.ReLU(),
        )
        
        combined_width = NUM_IMAGES * 4 + NUM_SCALARS
        linears = [
            nn.Linear(combined_width, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        ]
        for i in range(num_blocks):
            linears.append(ResBlock(channels, channels))
        linears.append(
            nn.Linear(channels, NUM_ACTIONS)
        )
        self.linears = nn.Sequential(*linears)

        if torch.cuda.is_available() and not self.force_cpu:
            self.cuda()

    def forward(self, images, scalars):
        x = self.convs(images)
        x = torch.flatten(x, start_dim=1)
        x = self.image_linears(x)
        x = torch.cat((x, scalars), dim=1)
        policy_scores = self.linears(x)
        return policy_scores

    def optimize(self, data, batch_size, optimizer, verbose=False):
        if len(data) < 2:
            # Need at least 2 data points for batch norm
            return None
        
        self.train()
        running_loss = 0
        pbar = tqdm(range(0, len(data) - batch_size + 1, batch_size), disable=(not verbose))
        for i in pbar:
            batch = data[i:i+batch_size]
            states, actions, rewards = zip(*batch)
            images, scalars = zip(*states)

            images_batch = np.stack(images)
            scalars_batch = np.stack(scalars)
            images_batch, scalars_batch = self.to_torch(images_batch, scalars_batch)

            actions_batch = np.array(actions)
            actions_onehot = np.zeros((actions_batch.shape[0], NUM_ACTIONS))
            actions_onehot[np.arange(actions_batch.shape[0]), actions_batch] = 1
            actions_onehot = torch.from_numpy(actions_onehot).type(torch.FloatTensor)
            rewards_batch = torch.from_numpy(np.array(rewards)).type(torch.FloatTensor)
            if torch.cuda.is_available() and not self.force_cpu:
                actions_onehot = actions_onehot.cuda()
                rewards_batch = rewards_batch.cuda()

            preds = self(images_batch, scalars_batch)
            log_probs = torch.nn.functional.log_softmax(preds, dim=1)
            log_probs_observed = torch.sum(log_probs * actions_onehot, dim=1)
            loss = -torch.sum(log_probs_observed * rewards_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_experiences = i + batch_size
            pbar.set_postfix_str("{:.3f}L".format(running_loss / num_experiences))
        pbar.close()

        num_batches = np.ceil(len(data) / batch_size)
        avg_loss = running_loss / num_batches
        return avg_loss

    def to_torch(self, images, scalars):
        images = torch.from_numpy(images).type(torch.FloatTensor)
        scalars = torch.from_numpy(scalars).type(torch.FloatTensor)
        if torch.cuda.is_available() and not self.force_cpu:
            images = images.cuda()
            scalars = scalars.cuda()
        return images, scalars