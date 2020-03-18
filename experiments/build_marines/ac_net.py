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

    def actor_criterion(self, policy_scores, actions_onehot, advantages):
        log_probs = torch.nn.functional.log_softmax(policy_scores, dim=1)
        log_probs_observed = torch.sum(log_probs * actions_onehot, dim=1)
        loss = -torch.sum(log_probs_observed * advantages)
        return loss

    def optimize(self, data, batch_size, optimizer, verbose=False):
        if len(data) < 2:
            # Need at least 2 data points for batch norm
            return None, None
        
        self.train()
        critic_loss = 0
        pbar = tqdm(range(0, len(data) - batch_size + 1, batch_size), disable=(not verbose))
        for i in pbar:
            batch = data[i:i+batch_size]
            states, _, values, _ = zip(*batch)
            images, scalars = zip(*states)

            images_batch = np.stack(images)
            scalars_batch = np.stack(scalars)
            values_batch = np.array(values)[:, np.newaxis]
            images_batch, scalars_batch, values_batch = self.to_torch(
                (images_batch, scalars_batch, values_batch))

            optimizer.zero_grad()
            _, vals = self(images_batch, scalars_batch)
            loss = self.critic_criterion(vals, values_batch)
            loss.backward()
            optimizer.step()

            critic_loss += loss.item()
            num_experiences = i + batch_size
            pbar.set_postfix_str("{:.3f}L".format(critic_loss / num_experiences))
        pbar.close()
        
        batched_advantages = self.get_batched_advantages(data, batch_size)
        self.train()
        actor_loss = 0
        # NOTE guarantee next state exists for advantage calculation
        pbar = tqdm(range(0, len(data) - batch_size, batch_size), disable=(not verbose))
        for i in pbar:
            batch = data[i:i+batch_size]
            states, actions, _, _ = zip(*batch)
            images, scalars = zip(*states)

            images_batch = np.stack(images)
            scalars_batch = np.stack(scalars)

            actions_batch = np.array(actions)
            actions_onehot = np.zeros((actions.shape[0], NUM_ACTIONS))
            actions_onehot[np.arange(actions.shape[0]), actions] = 1
            
            images_batch, scalars_batch, actions_onehot = self.to_torch(
                (images_batch, scalars_batch, actions_onehot))
            advantages_batch = batched_advantages[i:i+batch_size]

            optimizer.zero_grad()
            policy_scores, _ = self(images_batch, scalars_batch)
            loss = self.actor_criterion(policy_scores, actions_batch, advantages_batch)
            loss.backward()
            optimizer.step()

            actor_loss += loss.item()
            num_experiences = i + batch_size
            pbar.set_postfix_str("{:.3f}L".format(actor_loss / num_experiences))
        pbar.close()

        num_batches = np.ceil(len(data) / batch_size)
        avg_critic_loss = critic_loss / num_batches
        avg_actor_loss = actor_loss / num_batches
        return avg_critic_loss, avg_actor_loss

    def get_batched_advantages(self, data, batch_size):
        self.eval()
        values = np.zeros(len(data))
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            states, _, _, _ = zip(*batch)
            images, scalars = zip(*states)
            
            images_batch = np.stack(images)
            scalars_batch = np.stack(scalars)
            images_batch, scalars_batch = self.to_torch((images_batch, scalars_batch))

            _, vals = self(images_batch, scalars_batch)
            vals = vals.detach().cpu().numpy()
            values[i:i+batch_size] = vals
        
        # TODO calculate advantage
        for i in range(0, len(data) - batch_size, batch_size):
            batch = data[i:i+batch_size]
            _, _, _, terminals = zip(*batch)
            terminals = np.array(terminals)

            values[i:i+batch_size] -= values[i+1:i+batch_size+1]
        # TODO insert NaN, no advantage for last experience
        return

    @staticmethod
    def to_torch(arrays):
        def transform(x):
            x = torch.from_numpy(x).type(torch.FloatTensor)
            return x.cuda() if torch.cuda.is_available() else x
        return [transform(x) for x in arrays]