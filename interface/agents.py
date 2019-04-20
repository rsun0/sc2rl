class AbstractAgent:
    def __init__(self):
        pass

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

    def train(self, state, action, reward, next_state):
        '''
        Updates the agent with the experience of going from
        state to next_state when taking action
        '''
        raise NotImplementedError
