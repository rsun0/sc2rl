import json
import collections

import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, "../interface/")

from agent import Agent, Memory
from action_interface import BuildMarinesAction

NUM_ACTIONS = len(list(BuildMarinesAction))


class PolicyGradientMemory(Memory):
    def __init__(self, buffer_len, discount):
        self.experiences = collections.deque(maxlen=buffer_len)
        self.discount = discount
        self.current_trajectory = []

    def push(self, state, action, reward, done):
        state, env_state = state
        self.current_trajectory.append((state, action, env_state))
        
        if done:
            rewards = []
            r = reward
            for i in range(len(self.current_trajectory)):
                rewards.append(r)
                r *= self.discount
            rewards.reverse()

            states, actions, env_states = zip(*self.current_trajectory)

            trajectory = zip(states, actions, rewards, env_states)
            self.experiences.extend(trajectory)
            self.current_trajectory = []

    def get_data(self):
        return list(self.experiences)


class PolicyGradientAgent(Agent):
    def __init__(self, save_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_file = save_file

        self.train_count = 0

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
        data = self.memory.get_data()
        batch_size = run_settings.batch_size
        loss = self.model.optimize(
            data, batch_size, self.optimizer, self.settings.verbose)
        
        if self.train_count == 0:
            print('ITR', 'LOSS', sep='\t')
        print(f'{self.train_count:02d}', f'{100*loss:04.1f}', sep='\t')
        sys.stdout.flush()
        self.train_count += 1

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