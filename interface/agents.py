from memory import Memory
from pysc2.agents import base_agent
from pysc2.lib import actions, features


class AbstractAgent(base_agent.BaseAgent):
    def __init__(self):
        ''' 
        Stores s, a, r, s', d for each frame relavant to training or testing
        '''
        self.memory = Memory()
        
    def state_modifier(self, state):
        '''
        Returns an altered state for the agent based on the given state
        Output is same format as input to forward        
        '''
        raise NotImplementedError

    def sample(self, state):
        '''
        Returns an action chosen by the agent for the given state.
        May not be deterministic if the agent probabilistically
        chooses between multiple actions for the given state.
        '''
        raise NotImplementedError

    def forward(self, state):
        '''
        Returns the network output given the current state
        (usually a probability distribution of actions)
        '''
        raise NotImplementedError
        
    def train(self):
        '''
        Calls train_step until it has run through memory self.epochs times
        '''
        raise NotImplementedError
        
    def train_step(self, batch_size):
        '''
        Updates the agent with the experience of going from
        state to next_state when taking action
        '''
        raise NotImplementedError
        
    def action_space_converter(self, action):
        '''
        Takes in action, an output from self.sample
        Returns equivalent sc2env action
        '''
        raise NotImplementedError
        
