# https://github.com/tambetm/pommerman-baselines/blob/master/mcts/mcts_agent.py
import sys
sys.path.insert(0, "../interface/")

from agent import Agent

import argparse
import multiprocessing
from queue import Empty
import numpy as np
import time
import pickle

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants
import gym

import json
import time

NUM_AGENTS = 2
NUM_ACTIONS = len(constants.Action)
NUM_CHANNELS = 18
SAVE_FILE = 'mct.pickle'

total_time = {'obs_to_state': 0.0, 'env_step': 0.0, 'rollout': 0.0, 'search': 0.0}
total_frequency = {'obs_to_state': 0, 'env_step': 0, 'rollout': 0, 'search': 0} 

def argmax_tiebreaking(Q):
    # find the best action with random tie-breaking
    idx = np.flatnonzero(Q == np.max(Q))
    assert len(idx) > 0, str(Q)
    return np.random.choice(idx)


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


class MCTSAgent(BaseAgent, Agent):

    def __init__(self, agent_id=0, opponent=SimpleAgent(), *args, **kwargs):
        print('init')
        super(MCTSAgent, self).__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.env = self.make_env(opponent)
        self.reset_tree()
        self.num_episodes = 1
        self.mcts_iters = 3
        self.num_rollouts = 50
        self.mcts_c_puct = 1.0
        self.discount = 0.9
        self.temperature = 0.0

    def make_env(self, opponent):
        print('make_env')
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(opponent)

        return pommerman.make('OneVsOne-v0', agents)

    def reset_tree(self):
        # print('reset_tree')
        self.tree = {}

    def obs_to_state(self, obs):
        start_time = time.time()
        obs = obs[self.agent_id]

        board = obs['board']
        state = np.zeros((4, board.shape[0], board.shape[1]))
        state[0] = board
        state[1] = obs['bomb_life']
        state[2] = obs['bomb_moving_direction']
        state[3] = obs['flame_life']

        time_elapsed = time.time() - start_time
        total_time['obs_to_state'] += time_elapsed
        total_frequency['obs_to_state'] += 1
        
        state = state.astype(np.uint8)
        return state.tobytes()

    def search(self, root, num_iters, temperature=1):
        # print('search', root['step_count'])
        # remember current game state
        self.env._init_game_state = root
        self.env.set_json_info()

        root.pop('intended_actions')
        temp = self.env.get_json_info()
        temp.pop('intended_actions')
        assert str(root) == str(temp)

        self.env.training_agent = self.agent_id
        obs = self.env.reset()
        root_state = self.obs_to_state(obs)
        # print('root')

        for i in range(num_iters):
            # restore game state to root node
            obs = self.env.reset()

            # serialize game state
            temp = self.env.get_json_info()
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
                    probs = np.ones(NUM_ACTIONS) / NUM_ACTIONS

                    # use current rewards for values
                    rewards = self.env._get_rewards()
                    reward = rewards[self.agent_id]

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

        # reset env back where we were
        self.env.set_json_info()
        self.env._init_game_state = None

        # return action probabilities
        return self.tree[root_state].probs(self.temperature)

    def rollout(self):
        # print('rollout')
        # reset search tree in the beginning of each rollout
        # self.reset_tree()

        # guarantees that we are not called recursively
        # and episode ends when this agent dies
        self.env.training_agent = self.agent_id
        obs = self.env.reset()

        length = 0
        done = False
        my_actions = []
        while not done:
            root = self.env.get_json_info()
            # do Monte-Carlo tree search
            search_start_time = time.time()
            pi = self.search(root, self.mcts_iters, self.temperature)
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
            # print("Agent:", self.agent_id, "Step:", length, "Actions:", [constants.Action(a).name for a in actions], "Probs:", [round(p, 2) for p in pi], "Rewards:", rewards, "Done:", done)

        reward = rewards[self.agent_id]
        # Discount
        reward = reward * self.discount ** (length - 1)
        return length, reward, rewards, my_actions

    def act(self, obs, action_space):
        board_info = obs[0]
        scalar_info = obs[1]
        environment = obs[2] # 'json_info
        #print(environment)
        self.env._init_game_state = environment
        self.env.set_json_info()
        print('Number of nodes: ', len(self.tree))
        
        frequency = dict()
        avg_length = dict()
        avg_reward = dict()
        for i in range(self.num_rollouts):
            rollout_start_time = time.time()
            length, reward, _, my_actions = self.rollout()
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

        # pi = self.search(environment, self.mcts_iters, self.temperature)
        # action = np.random.choice(NUM_ACTIONS, p=pi)
        best_action = list(avg_reward.keys())[0]
        for action in avg_reward:
            if avg_reward[action] > avg_reward[best_action]:
                best_action = action
        print(avg_reward)
        print(avg_length)
        print('act', best_action)

        print('timing info')
        for action in total_time:
            print(action, total_time[action] / total_frequency[action])

        return best_action

    def _sample(self, state):
        return self.act(state, gym.spaces.discrete.Discrete(NUM_ACTIONS))

    def _forward(self, state):
        return self._sample(state)

    def state_space_converter(self, obs):
        to_use = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]
 
        board = obs['board'] # 0-4, 6-8, 10-11 [10 total]
        bomb_life = obs['bomb_life'] # 11
        bomb_moving_direction = obs['bomb_moving_direction'] #12
        flame_life = obs['flame_life'] #13

        state = np.zeros((13, board.shape[0], board.shape[1]))
        for i in range(len(to_use)):
            state[i] = (board == to_use[i]).astype(int)
        state[10] = bomb_life 
        state[11] = bomb_moving_direction 
        state[12] = flame_life 

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

        return state, scalars, obs['json_info']

    def action_space_converter(self, action):
        return action

    def train(self, run_settings):
        pass

    def train_step(self, batch_size):
        pass

    def save(self):
        print('Saving {} tree nodes'.format(len(self.tree)))
        with open(SAVE_FILE, 'wb') as f:
            pickle.dump(self.tree, f)

    def load(self):
        try:
            with open(SAVE_FILE, 'rb') as f:
                self.tree = pickle.load(f)
        except FileNotFoundError:
            pass
    
    def push_memory(self, state, action, reward, done):
        pass

def runner(id, num_episodes, fifo, _args):
    # make args accessible to MCTSAgent
    global args
    args = _args
    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS
    agent = MCTSAgent(agent_id=agent_id)

    data = []
    for i in range(num_episodes):
        # do rollout
        start_time = time.time()
        length, reward, rewards = agent.rollout()
        elapsed = time.time() - start_time
        # add data samples to log
        fifo.put((length, reward, rewards, agent_id, elapsed))


def profile_runner(id, num_episodes, fifo, _args):
    import cProfile
    command = """runner(id, num_episodes, fifo, _args)"""
    cProfile.runctx(command, globals(), locals(), filename=_args.profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile')
    parser.add_argument('--render', action="store_true", default=False)
    # runner params
    parser.add_argument('--num_episodes', type=int, default=400)
    parser.add_argument('--num_runners', type=int, default=4)
    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=10)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    # RL params
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=0)
    args = parser.parse_args()

    assert args.num_episodes % args.num_runners == 0, "The number of episodes should be divisible by number of runners"

    # use spawn method for starting subprocesses
    ctx = multiprocessing.get_context('spawn')

    # create fifos and processes for all runners
    fifo = ctx.Queue()
    for i in range(args.num_runners):
        process = ctx.Process(target=profile_runner if args.profile else runner, args=(i, args.num_episodes // args.num_runners, fifo, args))
        process.start()

    # do logging in the main process
    all_rewards = []
    all_lengths = []
    all_elapsed = []
    for i in range(args.num_episodes):
        # wait for a new trajectory
        length, reward, rewards, agent_id, elapsed = fifo.get()

        print("Episode:", i, "Reward:", reward, "Length:", length, "Rewards:", rewards, "Agent:", agent_id, "Time per step:", elapsed / length)
        all_rewards.append(reward)
        all_lengths.append(length)
        all_elapsed.append(elapsed)

    print("Average reward:", np.mean(all_rewards))
    print("Average length:", np.mean(all_lengths))
    print("Time per timestep:", np.sum(all_elapsed) / np.sum(all_lengths))