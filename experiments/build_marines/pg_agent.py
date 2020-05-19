import collections
import itertools
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
import scipy.special

import sys
sys.path.insert(0, "../interface/")

from agent import Agent, Memory
from action_interface import BuildMarinesAction
from custom_env import SCREEN_SIZE
from action_interface import NUM_ACTIONS

NUM_IMAGES = 2
NUM_SCALARS = 8


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
            self.experiences.extend(trajectory)
            self.scores.append(values[0])
            self.num_exp += len(self.current_trajectory)
            self.num_games += 1
            self.current_trajectory = []

    def get_shuffled_data(self):
        return np.random.permutation(list(self.experiences))

    def get_average_score(self):
        return np.mean(self.scores)

    def discard(self):
        self.current_trajectory = []


class PolicyGradientAgent(Agent):
    def __init__(self,
            init_temp=0.0,
            temp_steps=1,
            save_file=None,
            log_file=None,
            force_cpu=False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_file = save_file

        self.init_temp = init_temp
        self.temp_steps = temp_steps
        self.temp = init_temp
        self.force_cpu = force_cpu
        self.train_count = 0
        
        print('ITR\tTIME\t\tGAMES\tEXP\t\tTEMP\tLR\t\tLOSS\t\tSCORE', file=log_file)
        time = datetime.now().strftime('%H:%M:%S')
        print('start\t{time}'.format(time=time), file=log_file, flush=True)

    def _sample(self, state):
        probs = self._forward(state)
        if self.temp > 0:
            probs = scipy.special.softmax(probs / self.temp)
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
        
        if loss is not None:
            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.scheduler.step()

            avg_score = self.memory.get_average_score()
            time = datetime.now().strftime('%H:%M:%S')
            print('{itr:<3d}\t{time}\t{games:5d}\t{exp:8d}\t{tmp:6.4f}\t{lr:8.2e}\t{loss:8.4f}\t{score:5.1f}'
                .format(
                    itr=self.train_count,
                    time=time,
                    games=self.memory.num_games,
                    exp=self.memory.num_exp,
                    tmp=self.temp,
                    lr=curr_lr,
                    loss=loss,
                    score=avg_score),
                file=run_settings.log_file, flush=True)

        if self.temp > 0:
            self.temp -= (self.init_temp / self.temp_steps)
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
                if torch.cuda.is_available() and not self.force_cpu:
                    self.model.load_state_dict(
                        torch.load(self.save_file, map_location=torch.device('cuda'))
                    )
                else:
                    self.model.load_state_dict(
                        torch.load(self.save_file, map_location=torch.device('cpu'))
                    )
            except FileNotFoundError:
                print('No policy network save file found')

    def push_memory(self, state, action, reward, done):
        self.memory.push(state, action, reward, done)

    def notify_episode_crashed(self, run_settings):
        print('Episode crashed', file=run_settings.log_file, flush=True)
        self.memory.discard()

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