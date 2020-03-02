import collections

import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, "../interface/")

from agent import Agent, Memory
from action_interface import BuildMarinesAction
from custom_env import SCREEN_SIZE
from action_interface import NUM_ACTIONS

NUM_CHANNELS = 11


class PolicyGradientMemory(Memory):
    def __init__(self, buffer_len, discount):
        self.experiences = collections.deque(maxlen=buffer_len)
        self.discount = discount
        self.current_trajectory = []

    def push(self, state, action, reward, done):
        self.current_trajectory.append((state, action, reward))
        
        if done:
            states, actions, rewards = zip(*self.current_trajectory)
            
            values = []
            running_value = 0
            for i in range(len(rewards) - 1, -1, -1):
                running_value += rewards[i]
                values.append(running_value)
                running_value *= self.discount
            values.reverse()

            trajectory = zip(states, actions, values)
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
        preds = self.model(state[np.newaxis])
        probs = torch.nn.functional.softmax(preds, dim=1).detach().numpy()[0]
        return probs

    def train(self, run_settings):
        data = self.memory.get_data()
        batch_size = run_settings.batch_size
        loss = self.model.optimize(
            data, batch_size, self.optimizer, self.settings.verbose)
        
        if self.train_count == 0:
            print('ITR\tLOSS', file=run_settings.log_file)
        if loss is not None:
            print('{itr:02d}\t{loss:.4f}'.format(itr=self.train_count, loss=loss),
                file=run_settings.log_file)
        else:
            print('{itr:02d}\tNone'.format(itr=self.train_count),
                file=run_settings.log_file)
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

    def state_space_converter(self, raw_state):
        obs, cc_queue_len = raw_state
        state = np.zeros((NUM_CHANNELS, SCREEN_SIZE, SCREEN_SIZE), dtype=int)
        state_idx = 0

        features = [
            obs.observation.feature_screen.unit_type,
            obs.observation.feature_screen.unit_hit_points,
            obs.observation.feature_screen.unit_hit_points_ratio,
            obs.observation.player.minerals,
            obs.observation.player.food_used,
            obs.observation.player.food_cap,
            obs.observation.player.food_army,
            obs.observation.player.food_workers,
            obs.observation.player.army_count,
            obs.observation.game_loop.item(),
            cc_queue_len,
        ]

        for f in features:
            state[state_idx] = f
            state_idx += 1

        assert state_idx == state.shape[0], state_idx
        return state

    def action_space_converter(self, action):
        return action