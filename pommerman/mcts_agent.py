# https://github.com/tambetm/pommerman-baselines/blob/master/mcts/mcts_agent.py
import sys
sys.path.insert(0, "../interface/")

from agent import Agent, Memory

import argparse
import multiprocessing
from queue import Empty
import numpy as np
import torch
import time
import pickle
import collections

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants
import gym

import json
import time
from tqdm import tqdm

NUM_AGENTS = 2
NUM_ACTIONS = len(constants.Action)
NUM_CHANNELS = 20

total_time = {'obs_to_state': 0.0, 'env_step': 0.0, 'rollout': 0.0, 'search': 0.0}
total_frequency = {'obs_to_state': 0, 'env_step': 0, 'rollout': 0, 'search': 0} 

def argmax_tiebreaking(Q):
    # find the best action with random tie-breaking
    idx = np.flatnonzero(Q == np.max(Q))
    assert len(idx) > 0, str(Q)
    return np.random.choice(idx)


class MCTSMemory(Memory):
    def __init__(self, buffer_len, discount):
        self.experiences = collections.deque(maxlen=buffer_len)
        self.discount = discount
        self.current_trajectory = []

    def push(self, state, action, reward, done):
        state = state[0]
        self.current_trajectory.append( (state, action) )
        
        if done:
            rewards = []
            r = reward
            for i in range(len(self.current_trajectory)):
                rewards.append(r)
                r *= self.discount
            rewards.reverse()

            states, actions = zip(*self.current_trajectory)

            trajectory = zip(states, actions, rewards)
            self.experiences.extend(trajectory)
            self.current_trajectory = []

    def get_data(self):
        return list(self.experiences)


class MCTSNode(object):
    def __init__(self, p, mcts_c_puct):
        # values for 6 actions
        self.Q = np.zeros(NUM_ACTIONS)
        self.W = np.zeros(NUM_ACTIONS)
        self.N = np.zeros(NUM_ACTIONS, dtype=np.uint32)
        assert p.shape == (NUM_ACTIONS,)
        self.P = p
        self.mcts_c_puct = mcts_c_puct

    def action(self):
        U = self.mcts_c_puct * self.P * np.sqrt(np.sum(self.N)) / (1 + self.N)
        return argmax_tiebreaking(self.Q + U)

    def update(self, action, reward):
        self.W[action] += reward
        self.N[action] += 1
        self.Q[action] = self.W[action] / self.N[action]

    def probs(self, temperature=1):
        if temperature == 0:
            p = np.zeros(NUM_ACTIONS)
            p[argmax_tiebreaking(self.N)] = 1
            return p
        else:
            Nt = self.N ** (1.0 / temperature)
            return Nt / np.sum(Nt)


class MCTSAgent(Agent, BaseAgent):

    def __init__(self,
            mcts_iters,
            num_rollouts=1,
            discount=1.0,
            c=1.5,
            temp=1.0,
            agent_id=0,
            opponent=SimpleAgent(),
            tree_save_file=None,
            model_save_file=None,
            *args,
            **kwargs):
        super(MCTSAgent, self).__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.env = self.make_env(opponent)
        self.reset_tree()
        self.mcts_iters = mcts_iters
        self.num_rollouts = num_rollouts
        self.mcts_c_puct = c
        self.discount = discount
        self.temperature = temp

        self.tree_save_file = tree_save_file
        self.model_save_file = model_save_file

    def make_env(self, opponent):
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(opponent)

        return pommerman.make('OneVsOne-v0', agents)

    def reset_game(self, root):
        # remember current game state
        self.env._init_game_state = dict()
        for key in root:
            self.env._init_game_state[key] = root[key]
        self.env.set_json_info()

    def reset_tree(self):
        # print('reset_tree')
        self.tree = {}

    def obs_to_state(self, obs):
        start_time = time.time()
        obs = obs[self.agent_id]

        board = obs['board']
        # state = np.zeros((4, board.shape[0], board.shape[1]))
        # state[0] = board
        # state[1] = obs['bomb_life']
        # state[2] = obs['bomb_moving_direction']
        # state[3] = obs['flame_life']
        state = board

        time_elapsed = time.time() - start_time
        total_time['obs_to_state'] += time_elapsed
        total_frequency['obs_to_state'] += 1
        
        state = state.astype(np.uint8)
        return state.tobytes()

    def search(self, root, num_iters, temperature=1):
        # print('search', root['step_count'])

        self.env.training_agent = self.agent_id
        obs = self.env.get_observations()
        root_state = self.obs_to_state(obs)

        for i in range(num_iters):
            # restore game state to root node
            self.reset_game(root)

            # serialize game state
            obs = self.env.get_observations()
            state = self.obs_to_state(obs)

            trace = []
            done = False
            while not done:
                if state in self.tree:
                    node = self.tree[state]
                    # choose actions based on Q + U
                    action = node.action()
                    trace.append((node, action))
                else:
                    # use unfiform distribution for probs
                    # probs = np.ones(NUM_ACTIONS) / NUM_ACTIONS

                    # Use policy network to initialize probs
                    model_in, _ = self.state_space_converter(
                        self.env.get_observations()[self.agent_id])
                    self.model.eval()
                    pi_scores, values = self.model(model_in[np.newaxis])
                    probs = torch.nn.functional.softmax(pi_scores, dim=1).detach().numpy()[0]

                    if self.env._get_done():
                        reward = self.env._get_rewards()[self.agent_id]
                    else:
                        # Use critic network for values
                        reward = torch.tanh(values).detach().numpy()[0]

                    # add new node to the tree
                    self.tree[state] = MCTSNode(probs, self.mcts_c_puct)

                    # stop at leaf node
                    break

                # ensure we are not called recursively
                assert self.env.training_agent == self.agent_id
               
                # make other agents act
                actions = self.env.act(obs)
                # add my action to list of actions
                actions.insert(self.agent_id, action)
                # step environment forward
                env_start_time = time.time()
                obs, rewards, done, info = self.env.step(actions)

                env_time_elapsed = time.time() - env_start_time
                total_time['env_step'] += env_time_elapsed
                total_frequency['env_step'] += 1
                reward = rewards[self.agent_id]

                # fetch next state
                state = self.obs_to_state(obs)


            # update tree nodes with rollout results
            for node, action in reversed(trace):
                node.update(action, reward)
                reward *= self.discount

        # return action probabilities
        return self.tree[root_state].probs(self.temperature)

    def rollout(self):
        # print('rollout')
        # reset search tree in the beginning of each rollout
        self.reset_tree()

        # guarantees that we are not called recursively
        # and episode ends when this agent dies
        self.env.training_agent = self.agent_id
        obs = self.env.get_observations()

        length = 0
        done = False
        my_actions = []
        my_policies = []
        while not done:
            root = self.env.get_json_info()
            root.pop('intended_actions')
            # do Monte-Carlo tree search
            search_start_time = time.time()

            # load original state
            # print(root)
            self.reset_game(root) 

            # print(done, type(done))
            # self.env.render()
            pi = self.search(root, self.mcts_iters, self.temperature)

            my_policies.append(pi)

            # reset env back where we were
            self.reset_game(root)

            search_time_elapsed = time.time() - search_start_time
            total_time['search'] += search_time_elapsed
            total_frequency['search'] += 1

            # sample action from probabilities
            action = np.random.choice(NUM_ACTIONS, p=pi)
            my_actions.append(action)

            # ensure we are not called recursively
            assert self.env.training_agent == self.agent_id

            # make other agents act
            actions = self.env.act(obs)

            # add my action to list of actions
            actions.insert(self.agent_id, action)

            # step environment
            env_start_time = time.time()

            obs, rewards, done, info = self.env.step(actions)

            env_time_elapsed = time.time() - env_start_time
            total_time['env_step'] += env_time_elapsed
            total_frequency['env_step'] += 1

            assert self == self.env._agents[self.agent_id]
            length += 1

        reward = rewards[self.agent_id]
        # Discount
        reward = reward * self.discount ** (length - 1)
        return length, reward, rewards, my_actions, my_policies

    def act(self, obs, action_space):
        state = obs[0]
        environment = obs[1] # json_info
        
        frequency = dict()
        avg_length = dict()
        avg_reward = dict()

        for i in range(self.num_rollouts):
            rollout_start_time = time.time()

            self.reset_game(environment)

            length, reward, _, my_actions, my_policies = self.rollout()
            rollout_time_elapsed = time.time() - rollout_start_time
            total_time['rollout'] += rollout_time_elapsed
            total_frequency['rollout'] += 1
            a = my_actions[0]

            if a in frequency:
                avg_length[a] = (frequency[a])/(frequency[a] + 1) * avg_length[a] + length / (frequency[a] + 1)
                avg_reward[a] = (frequency[a])/(frequency[a] + 1) * avg_reward[a] + reward / (frequency[a] + 1)
                frequency[a] += 1
            else:
                avg_length[a] = length
                avg_reward[a] = reward
                frequency[a] = 1

        best_action = max(avg_reward, key=avg_reward.get)

        # print('timing info')
        # for action in total_time:
        #     print(action, total_time[action] / total_frequency[action])

        return best_action

    def _sample(self, state):
        return self.act(state, gym.spaces.discrete.Discrete(NUM_ACTIONS))

    def _forward(self, state):
        return self._sample(state)

    def state_space_converter(self, obs):
        board = obs['board']
        state = np.zeros((NUM_CHANNELS, board.shape[0], board.shape[1]))
        state_idx = 0

        board_indices = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]
        for b in board_indices:
            state[state_idx] = (board == b).astype(int)
            state_idx += 1
        additional_images = ['bomb_blast_strength', 'bomb_life',
            'bomb_moving_direction', 'flame_life']
        for im in additional_images:
            state[state_idx] = obs[im]
            state_idx += 1

        scalar_items = ['ammo', 'blast_strength', 'can_kick']
        # array of dictionaries as a string
        agents = obs['json_info']['agents']
       
        i = agents.find('}')
        agent1 = json.loads(obs['json_info']['agents'][1:i+1])
        agent2 = json.loads(obs['json_info']['agents'][i+2:-1]) 

        for agent in [agent1, agent2]:
            for scalar_item in scalar_items:
                state[state_idx] = int(agent[scalar_item])
                state_idx += 1

        assert state_idx == state.shape[0], state_idx
        return state, obs['json_info']

    def action_space_converter(self, action):
        return action

    def train(self, run_settings):
        self.model.train()
        data = self.memory.get_data()
        pbar = tqdm(range(0, len(data), run_settings.batch_size))
        running_loss = 0
        for i in pbar:
            batch = data[i:i+run_settings.batch_size]
            if len(batch) == 1:
                # Batch norm will fail
                break
            states, actions, rewards = zip(*batch)

            states_batch = np.stack(states)
            actions_batch = np.array(actions)
            rewards_batch = torch.from_numpy(np.array(rewards))

            actions_onehot = np.zeros((actions_batch.shape[0], NUM_ACTIONS))
            actions_onehot[np.arange(actions_batch.shape[0]), actions_batch] = 1
            actions_onehot = torch.from_numpy(actions_onehot).type(torch.FloatTensor)

            preds, _ = self.model(states_batch)
            log_probs = torch.nn.functional.log_softmax(preds, dim=1)
            log_probs_observed = torch.sum(log_probs * actions_onehot, dim=1)
            loss = -torch.sum(log_probs_observed * rewards_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_experiences = i + run_settings.batch_size
            pbar.set_postfix_str("{:.3f}L".format(running_loss / num_experiences))

    def train_step(self, batch_size):
        pass

    def save(self):
        if self.tree_save_file:
            print('Saving {} tree nodes'.format(len(self.tree)))
            with open(self.tree_save_file, 'wb') as f:
                pickle.dump(self.tree, f)
        if self.model_save_file:
            print('Saving policy network')
            torch.save(self.model.state_dict(), self.model_save_file)

    def load(self):
        if self.tree_save_file:
            try:
                with open(self.tree_save_file, 'rb') as f:
                    self.tree = pickle.load(f)
            except FileNotFoundError:
                print('No tree save file found')
        if self.model_save_file:
            try:
                self.model.load_state_dict(torch.load(self.model_save_file))
            except FileNotFoundError:
                print('No policy network save file found')
    
    def push_memory(self, state, action, reward, done):
        self.memory.push(state, action, reward, done)