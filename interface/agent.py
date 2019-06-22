class Agent():
    def __init__(self, model, settings, memory):
        self.model = model
        self.settings = settings
        self.memory = memory
        
        # Instantiate optimizer
        self.optimizer = self.settings.optimizer(
            params=self.model.parameters(), lr=self.settings.learning_rate)
    
    def sample(self, state):
        """
        Calls _sample after wrapping with state_space_converter
        and action_space_converter
        """
        internal_state = self.state_space_converter(state)
        internal_action = self._sample(internal_state)
        return self.action_space_converter(internal_action)

    def forward(self, state):
        """
        Calls _forward after wrapping with state_space_converter
        """
        internal_state = self.state_space_converter(state)
        return self._forward(internal_state)

    def state_space_converter(self, state):
        """
        Returns an altered state for the agent based on the given state
        Output is same format as input to forward        
        """
        raise NotImplementedError
        
    def action_space_converter(self, action):
        """
        Takes in action, an output from self.sample
        Returns equivalent CustomEnvironment action
        """
        raise NotImplementedError

    def _sample(self, state):
        """
        Returns an action chosen by the agent for the given state.
        May not be deterministic if the agent probabilistically
        chooses between multiple actions for the given state.
        """
        raise NotImplementedError

    def _forward(self, state):
        """
        Returns the network output given the current state
        (usually a probability distribution of actions)
        """
        raise NotImplementedError
        
    def train(self):
        """
        Calls train_step until it has run through memory self.epochs times
        """
        raise NotImplementedError
        
    def train_step(self, batch_size):
        """
        Updates the agent with the experience of going from
        state to next_state when taking action
        """
        raise NotImplementedError

    def save(self):
        """
        Saves model and necessary data
        """
        raise NotImplementedError
        

class Model():
    def parameters(self):
        raise NotImplementedError()


class AgentSettings():
    def __init__(self, optimizer, learning_rate,
            epsilon_max, epsilon_min, epsilon_duration):
        """
        :param optimizer: A class from torch.optim (instantiated later)
        :param learning_rate: The learning rate for the network
        :param epsilon_max: The starting epsilon
        :param epsilon_min: The final epsilon
        :param epsilon_duration: The number of frames to reach the final epsilon
        """
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_duration = epsilon_duration

    def get_epsilon(self, frame_num):
        """
        Computes the epsilon for a given frame based
        on epsilon_max, epsilon_min, and epsilon_duration
        """
        progress = frame_num / self.epsilon_duration
        reduction = -(self.epsilon_max - self.epsilon_min) * progress
        return max(self.epsilon_min, self.epsilon_max + reduction)


class Memory():
    def __init__(self):
        raise NotImplementedError

    def push(self, state, action, reward, done):
        raise NotImplementedError