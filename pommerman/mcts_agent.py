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
        self.mcts_c_puct = c
        self.discount = discount
        self.temperature = temp

        self.tree_save_file = tree_save_file
        self.model_save_file = model_save_file

        self.train_count = 0

    def make_env(self, opponent):
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(opponent)

        env = pommerman.make('OneVsOne-v0', agents)
        env.set_training_agent(self.agent_id)
        return env

    @staticmethod
    def set_state(env, state):
        env._init_game_state = state
        env.set_json_info()

    def reset_game(self, root):
        self.set_state(self.env, root)

    def reset_tree(self):
        # print('reset_tree')
        self.tree = {}

    def obs_to_state(self, obs):
        start_time = time.time()
        obs = obs[self.agent_id]

        state, _ = self.state_space_converter(obs)

        time_elapsed = time.time() - start_time
        total_time['obs_to_state'] += time_elapsed
        total_frequency['obs_to_state'] += 1
        
        state = state.astype(np.uint8)
        return state.tobytes()

    def search(self, root, num_iters, temperature=1):
        # print('search', root['step_count'])

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
                    # Use policy network to initialize probs
                    model_in, _ = self.state_space_converter(obs[self.agent_id])
                    self.model.eval()
                    pi_scores, values = self.model(model_in[np.newaxis])
                    probs = torch.nn.functional.softmax(pi_scores, dim=1).detach().numpy()[0]

                    if self.env._get_done():
                        reward = self.env._get_rewards()[self.agent_id]
                    else:
                        # Use critic network for values
                        reward = values.detach().numpy()[0]

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

    def act(self, obs, action_space):
        # print('Number of nodes: ', len(self.tree))
        _, env_state = obs

        self.reset_tree()
        self.reset_game(env_state)
        pi = self.search(env_state, self.mcts_iters, self.temperature)
        action = np.random.choice(NUM_ACTIONS, p=pi)
        return action

    def _sample(self, state):
        return self.act(state, gym.spaces.discrete.Discrete(NUM_ACTIONS))

    def _forward(self, state):
        return self._sample(state)

    @staticmethod
    def state_space_converter(obs):
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
        agents = json.loads(obs['json_info']['agents'])

        for agent in agents:
            for scalar_item in scalar_items:
                state[state_idx] = int(agent[scalar_item])
                state_idx += 1

        assert state_idx == state.shape[0], state_idx
        return state, obs['json_info']

    def action_space_converter(self, action):
        return action

    def train(self, run_settings):
        data = self.memory.get_data()
        batch_size = run_settings.batch_size
        c_loss, c_acc, a_loss, a_acc = self.model.optimize(
            data, batch_size, self.optimizer, self.env)

        if self.train_count == 0:
            print('ITR', 'C_ACC', 'C_LOSS', 'A_ACC', 'A_LOSS', sep='\t')
        print(f'{self.train_count:02d}',
            f'{100*c_acc:04.1f}\t{c_loss:04.3f}',
            f'{100*a_acc:04.1f}\t{a_loss:04.3f}',
            sep='\t')
        sys.stdout.flush()
        self.train_count += 1

    def train_step(self, batch_size):
        pass

    def save(self):
        # if self.tree_save_file:
        #     print('Saving {} tree nodes'.format(len(self.tree)))
        #     with open(self.tree_save_file, 'wb') as f:
        #         pickle.dump(self.tree, f)
        if self.model_save_file:
            if self.settings.verbose:
                print('Saving policy network')
            torch.save(self.model.state_dict(), self.model_save_file)

    def load(self):
        # if self.tree_save_file:
        #     try:
        #         with open(self.tree_save_file, 'rb') as f:
        #             self.tree = pickle.load(f)
        #     except FileNotFoundError:
        #         print('No tree save file found')
        if self.model_save_file:
            try:
                self.model.load_state_dict(torch.load(self.model_save_file))
            except FileNotFoundError:
                print('No policy network save file found')
    
    def push_memory(self, state, action, reward, done):
        self.memory.push(state, action, reward, done)