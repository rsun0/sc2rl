import sys
sys.path.insert(0, "../interface/")

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from agent import Model
from ac_agent import NUM_IMAGES, NUM_SCALARS
from policy_net import ResBlock
from custom_env import SCREEN_SIZE
from action_interface import NUM_ACTIONS


class ActorCriticNet(nn.Module, Model):
    def __init__(self,
            num_blocks=2,
            channels=32):
        super().__init__()

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
        self.linears = nn.Sequential(*linears)

        self.actor = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, NUM_ACTIONS),
        )

        self.critic = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

        self.actor_criterion = nn.CrossEntropyLoss()
        self.critic_criterion = nn.MSELoss()
        
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, images, scalars):
        x = self.convs(images)
        x = torch.flatten(x, start_dim=1)
        x = self.image_linears(x)
        x = torch.cat((x, scalars), dim=1)
        x = self.linears(x)

        policy_scores = self.actor(x)
        vals = self.critic(x)
        return policy_scores, vals

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
            if torch.cuda.is_available():
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

        # TODO use V(s+1) - V(s) as the advantage estimate

    @staticmethod
    def to_torch(images, scalars):
        images = torch.from_numpy(images).type(torch.FloatTensor)
        scalars = torch.from_numpy(scalars).type(torch.FloatTensor)
        if torch.cuda.is_available():
            images = images.cuda()
            scalars = scalars.cuda()
        return images, scalars