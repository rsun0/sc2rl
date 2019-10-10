# https://github.com/tambetm/pommerman-baselines/blob/master/mcts/mcts_agent.py
import sys
sys.path.insert(0, "../interface/")

from agent import Agent

import argparse
import multiprocessing
from queue import Empty
import numpy as np
import time

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants
import gym


NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action)
NUM_CHANNELS = 18


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

    def __init__(self, agent_id=0, *args, **kwargs):
        print('init')
        super(MCTSAgent, self).__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.env = self.make_env()
        self.reset_tree()
        self.num_episodes = 4 #400
        self.mcts_iters = 2 #10
        self.mcts_c_puct = 1.0
        self.discount =0.99
        self.temperature = 0.0

    def make_env(self):
        print('make_env')
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent())

        return pommerman.make('PommeFFACompetition-v0', agents)

    def reset_tree(self):
        print('reset_tree')
        self.tree = {}

    def search(self, root, num_iters, temperature=1):
        print('search')
        # remember current game state
        self.env._init_game_state = root
        self.env.set_json_info()

        root.pop('intended_actions')
        temp = self.env.get_json_info()
        temp.pop('intended_actions')
        assert str(root) == str(temp)

        self.env.training_agent = self.agent_id
        str_root = str(root)
        print('root')

        for i in range(num_iters):
            # restore game state to root node
            obs = self.env.reset()

            # serialize game state
            temp = self.env.get_json_info()
            temp.pop('intended_actions')
            state = str(temp)

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
                obs, rewards, done, info = self.env.step(actions)
                reward = rewards[self.agent_id]

                # fetch next state
                state = str(self.env.get_json_info())

            # update tree nodes with rollout results
            for node, action in reversed(trace):
                node.update(action, reward)
                reward *= self.discount

        # reset env back where we were
        self.env.set_json_info()
        self.env._init_game_state = None

        # return action probabilities
        return self.tree[str_root].probs(self.temperature)

    def rollout(self):
        print('rollout')
        # reset search tree in the beginning of each rollout
        self.reset_tree()

        # guarantees that we are not called recursively
        # and episode ends when this agent dies
        self.env.training_agent = self.agent_id
        obs = self.env.reset()

        length = 0
        done = False
        while not done:
            root = self.env.get_json_info()
            # do Monte-Carlo tree search
            pi = self.search(root, self.mcts_iters, self.temperature)
            # sample action from probabilities
            action = np.random.choice(NUM_ACTIONS, p=pi)

            # ensure we are not called recursively
            assert self.env.training_agent == self.agent_id
            # make other agents act
            actions = self.env.act(obs)
            # add my action to list of actions
            actions.insert(self.agent_id, action)
            # step environment
            obs, rewards, done, info = self.env.step(actions)
            assert self == self.env._agents[self.agent_id]
            length += 1
            print("Agent:", self.agent_id, "Step:", length, "Actions:", [constants.Action(a).name for a in actions], "Probs:", [round(p, 2) for p in pi], "Rewards:", rewards, "Done:", done)

        reward = rewards[self.agent_id]
        return length, reward, rewards

    def act(self, obs, action_space):
        print('act')
        environment = obs['json_info']
        pi = self.search(environment, self.mcts_iters, self.temperature)
        action = np.random.choice(NUM_ACTIONS, p=pi)

        return action

    def _sample(self, state):
        return self.act(state, gym.spaces.discrete.Discrete(NUM_ACTIONS))

    def _forward(self, state):
        return self._sample(state)

    def state_space_converter(self, state):
        return state

    def action_space_converter(self, action):
        return action

    def train(self, run_settings):
        pass

    def train_step(self, batch_size):
        pass

    def save(self):
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