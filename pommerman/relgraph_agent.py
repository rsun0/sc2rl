import sys
sys.path.insert(0, "../interface/")

import numpy as np

from agent import Agent, Memory

NUM_CHANNELS = 18


class RelGraphMemory(Memory):
    def __init__(self, buffer_len, discount):
        self.experiences = collections.deque(maxlen=buffer_len)
        self.discount = discount
        self.current_trajectory = []

    def push(self, state, action, reward, done):
        pass


class RelGraphAgent(Agent):
    def __init__(self,
        num_agents=2,
        board_size=8,
        init_eps=0.0001,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.init_eps = init_eps
        num_objects = board_size ** 2
        self.graph_dim = (num_agents, num_agents + num_objects)

        self.reset()

    def reset(self):
        self.prev_state = None
        self.prev_action = None
        self.prev_graph = np.random.random(self.graph_dim).astype('float32') + self.init_eps

    def state_space_converter(self, obs):
        board = obs['board']
        state = np.zeros((NUM_CHANNELS, board.shape[0], board.shape[1]), dtype='float32')
        state_idx = 0

        board_indices = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]
        for b in board_indices:
            state[state_idx] = (board == b).astype(int)
            state_idx += 1
        
        additional_images = [
            'bomb_blast_strength',
            'bomb_life',
            'bomb_moving_direction',
            'flame_life',
            'ammo',
            'blast_strength',
            'can_kick',
        ]
        for im in additional_images:
            state[state_idx] = obs[im]
            state_idx += 1

        state[state_idx] = self.prev_action if self.prev_action is not None else 0
        state_idx += 1

        assert state_idx == state.shape[0], state_idx
        return state

    def action_space_converter(self, personal_action):
        raise NotImplementedError

    def _sample(self, state):
        raise NotImplementedError

    def _forward(self, state):
        raise NotImplementedError

    def train(self, run_settings):
        raise NotImplementedError

    def train_step(self, batch_size):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def push_memory(self, state, action, reward, done):
        raise NotImplementedError
