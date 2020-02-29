import json

import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, "../interface/")

from agent import Agent
from action_interface import BuildMarinesAction

NUM_ACTIONS = len(list(BuildMarinesAction))


class PolicyGradientAgent(Agent):
    def __init__(self, save_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_file = save_file

    def _sample(self, state):
        probs = self._forward(state)
        action = np.random.choice(NUM_ACTIONS, p=probs)
        return action

    def _forward(self, state):
        self.model.eval()
        images, scalars = state
        preds = self.model((images[np.newaxis], scalars[np.newaxis]))
        probs = torch.nn.functional.softmax(preds, dim=1).detach().numpy()[0]
        return probs

    def train(self, run_settings):
        batch_size = run_settings.batch_size
        self.model.train()
        data = self.memory.get_data()
        running_loss = 0
        pbar = tqdm(range(0, len(data) - batch_size + 1, batch_size))
        for i in pbar:
            batch = data[i:i+batch_size]
            states, actions, rewards = zip(*batch)
            images, scalars = zip(*states)

            images_batch = np.stack(images)
            scalars_batch = np.stack(scalars)
            actions_batch = np.array(actions)
            rewards_batch = torch.from_numpy(np.array(rewards))

            actions_onehot = np.zeros((actions_batch.shape[0], NUM_ACTIONS))
            actions_onehot[np.arange(actions_batch.shape[0]), actions_batch] = 1
            actions_onehot = torch.from_numpy(actions_onehot)

            preds = self.model((images_batch, scalars_batch))
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
        # Throw away used experiences?
        # self.experiences = []

    def train_step(self, batch_size):
        pass
    
    def save(self):
        if self.save_file is not None:
            print('Saving policy network')
            torch.save(self.model.state_dict(), self.save_file)

    def load(self):
        if self.save_file is not None:
            try:
                self.model.load_state_dict(torch.load(self.save_file))
            except FileNotFoundError:
                print('No policy network save file found')

    def push_memory(self, state, action, reward, done):
        self.memory.push(state, action, reward, done)

    # TODO change state converter, check paper
    def state_space_converter(self, obs):
        # feature screen unit_type
        # feature screen build_progress
        
        # minerals
        # food used
        # food cap
        # food army
        # food workers
        # idle worker count
        # army_count

        # build queue
        return obs

    def action_space_converter(self, action):
        return action