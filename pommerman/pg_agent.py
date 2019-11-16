import json

import torch
import numpy as np

import sys
sys.path.insert(0, "../interface/")

from agent import Agent

class PolicyGradientAgent(Agent):
    def __init__(self, save_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_file = save_file

    def _sample(self, state):
        probs = self._forward(state)
        action = np.random.choice(6, p=probs)
        return action

    def _forward(self, state):
        self.model.eval()
        images, scalars = state
        preds = self.model((images[np.newaxis], scalars[np.newaxis]))
        probs = torch.nn.functional.softmax(preds, dim=1).detach().numpy()[0]
        return probs

    def train(self, run_settings):
        self.model.train()
        data = self.memory.get_data()
        for i in range(0, len(data), run_settings.batch_size):
            batch = data[i:i+run_settings.batch_size]
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
            print('Log probs for experienced actions: ', log_probs_observed)
            print('Rewards: ', rewards_batch)
            loss = -torch.sum(log_probs_observed * rewards_batch)
            print('Loss: ', loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # Throw away used experiences?
        # self.experiences = []

    def train_step(self, batch_size):
        pass
    
    def save(self):
        print('Saving policy network')
        torch.save(self.model.state_dict(), self.save_file)

    def load(self):
        try:
            self.model.load_state_dict(torch.load(self.save_file))
        except FileNotFoundError:
            print('No policy network save file found')

    def push_memory(self, state, action, reward, done):
        self.memory.push(state, action, reward, done)

    def state_space_converter(self, obs):
        to_use = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]
 
        board = obs['board'] # 0-4, 6-8, 10-11 [10 total]
        bomb_life = obs['bomb_life'] # 11
        bomb_moving_direction = obs['bomb_moving_direction'] #12
        flame_life = obs['flame_life'] #13

        images = np.zeros((13, board.shape[0], board.shape[1]))
        for i in range(len(to_use)):
            images[i] = (board == to_use[i]).astype(int)
        images[10] = bomb_life 
        images[11] = bomb_moving_direction 
        images[12] = flame_life 

        scalars = []
        scalar_items = ['ammo', 'blast_strength', 'can_kick']
        agents = obs['json_info']['agents'] # array of dictionaries as a string
       
        i = agents.find('}')
        agent1 = json.loads(obs['json_info']['agents'][1:i+1])
        agent2 = json.loads(obs['json_info']['agents'][i+2:-1]) 

        for agent in [agent1, agent2]:
            for scalar_item in scalar_items:
                scalars.append(agent[scalar_item])

        scalars = np.array(scalars)

        return images, scalars

    def action_space_converter(self, action):
        return action