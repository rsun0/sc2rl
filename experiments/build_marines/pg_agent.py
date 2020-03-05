import collections
import itertools

import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, "../interface/")

from agent import Agent, Memory
from action_interface import BuildMarinesAction
from custom_env import SCREEN_SIZE
from action_interface import NUM_ACTIONS

NUM_IMAGES = 2
NUM_SCALARS = 8
MEM_SKIP = 4


class PolicyGradientMemory(Memory):
    def __init__(self, buffer_len, discount, averaging_window):
        self.experiences = collections.deque(maxlen=buffer_len)
        self.scores = collections.deque(maxlen=averaging_window)
        self.discount = discount
        self.current_trajectory = []
        self.num_games = 0
        self.num_exp = 0

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
            # Only save 1 out of MEM_SKIP experiences
            trajectory = itertools.islice(trajectory, None, None, MEM_SKIP)
            self.experiences.extend(trajectory)

            self.scores.append(values[0])
            self.num_exp += len(self.current_trajectory) // MEM_SKIP
            self.num_games += 1
            self.current_trajectory = []

    def get_shuffled_data(self):
        return np.random.permutation(list(self.experiences))

    def get_average_score(self):
        return np.mean(self.scores)


class PolicyGradientAgent(Agent):
    def __init__(self, save_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_file = save_file

        self.train_count = 0

    def _sample(self, state):
        # TODO add exploration?
        probs = self._forward(state)
        action = np.random.choice(NUM_ACTIONS, p=probs)
        return action

    def _forward(self, state):
        images, scalars = state
        images, scalars = self.model.to_torch(images[np.newaxis], scalars[np.newaxis])

        self.model.eval()
        preds = self.model(images, scalars)
        probs = torch.nn.functional.softmax(preds, dim=1).detach().cpu().numpy()[0]
        return probs

    def train(self, run_settings):
        data = self.memory.get_shuffled_data()
        batch_size = run_settings.batch_size
        loss = self.model.optimize(
            data, batch_size, self.optimizer, self.settings.verbose)
        
        if self.train_count == 0:
            print('ITR\tGAMES\tEXP\t\tLOSS\t\tSCORE', file=run_settings.log_file)
        if loss is not None:
            avg_score = self.memory.get_average_score()
            print('{itr:<3d}\t{games:4d}\t{exp:8d}\t{loss:8.4f}\t{score:5.1f}'
                .format(
                    itr=self.train_count,
                    games=self.memory.num_games,
                    exp=self.memory.num_exp,
                    loss=loss,
                    score=avg_score),
                file=run_settings.log_file, flush=True)
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

        images = np.empty((NUM_IMAGES, SCREEN_SIZE, SCREEN_SIZE), dtype=int)
        images[0] = obs.observation.feature_screen.unit_type
        images[1] = obs.observation.feature_screen.unit_hit_points
        # images[2] = obs.observation.feature_screen.unit_hit_points_ratio

        scalars = np.empty((NUM_SCALARS), dtype=int)
        scalars[0] = obs.observation.player.minerals
        scalars[1] = obs.observation.player.food_used
        scalars[2] = obs.observation.player.food_cap
        scalars[3] = obs.observation.player.food_army
        scalars[4] = obs.observation.player.food_workers
        scalars[5] = obs.observation.player.army_count
        scalars[6] = obs.observation.game_loop.item()
        scalars[7] = cc_queue_len

        return images, scalars

    def action_space_converter(self, action):
        return action