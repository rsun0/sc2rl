import sys
sys.path.insert(0, "../interface/")

from agent import Agent, Memory

import numpy as np
import torch

import pommerman
from pommerman import constants
import gym

NUM_AGENTS = 2
NUM_ACTIONS = len(constants.Action)


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        # TODO use env to determine terminal
        self.terminal = None
        # TODO make helper function to invalidate useless moves
        self.valid_moves = GoGame.get_valid_moves(state)

        # TODO use helper function
        self.actionsize = GoGame.get_action_size(state)
        self.canon_children = np.empty(self.actionsize, dtype=object)
        self.visits = 0

        self.prior_pi = None
        self.post_vals = []

        self.parent = parent