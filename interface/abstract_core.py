"""

Created by Michael McGuire and Ray Sun

Purpose: a generalized interface that can train arbitrary models and configurations.
To create a new agent for a new map or purpose,
we want to only have to define a concise set of new functions.
These functions and settings are passed into the Experiment constructor.

"""

class Experiment:
    def __init__(self, agents, model, custom_env, optim_settings, run_settings):
        self.agents = agents
        self.model = model
        self.custom_env = custom_env
        self.optim_settings = optim_settings
        self.run_settings = run_settings

    def train():
        """
        Trains the agent(s) using the custom environment
        """
        # Instantiate optimizer
        optimizer = self.optim_settings.optimizer(
            params=self.model.parameters(), lr=self.optim_settings.learning_rate)
        
        train_length = max([self.agents[i].train_length for i in range(len(self.agents))])
        
        env_obs = self.custom_env.reset()
        
        for frame in range(train_length):
            
            actions = []
            for i in range(len(self.agents))
                
                agent = self.agents[i]
                
                if frame % agent.train_every == 0:
                    agent.train()
                
                agent_state = env_obs[i].observation
                agent_action, agent_modified_state = agent.sample(agent_state)
                converted_action = agent.action_space_converter(agent_action)
                
                agent_states[i] = agent_modified_state
                agent_actions[i] = converted_action
                
            env_obs = self.custom_env.step(agent_actions)
            agent_rewards = [env_obs[i].reward for i in range(len(env_obs))]
            agent_dones = [env_obs[i].last() for i in range(len(env_obs))]
            
            if (frame != 0):
                for i in range(len(self.agents)):
                    ### Push memory
                    agent_modified_state = agent_states[i]
                    agent_action = agent_actions[i]
                    agent_reward = agent_rewards[i]
                    agent_done = agent_dones[i]
                    agent.memory.push([agent_modified_space, agent_action,
                        agent_reward, agent_done])
                    
            if (np.array(agent_dones) == False).all():
                env_obs = self.custom_env.reset()
                agent_states = [None for i in range(len(self.agents))]
                agent_actions = [None for i in range(len(self.agents))]
                agent_rewards = [0 for i in range(len(self.agents))]
                agent_dones = [False for i in range(len(self.agents))]


class Model:
    def parameters(self):
        raise NotImplementedError()

class CustomEnvironment:
    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class OptimizerSettings:
    def __init__(self, optimizer, learning_rate):
        """
        :param optimizer: A class from torch.optim (instantiated later)
        """
        # optimizer should be a class, will be instantiated later
        self.optimizer = optimizer
        self.learning_rate = learning_rate


class RunSettings:
    def __init__(self, num_epochs, batch_size):
        self.num_epochs = num_epochs
        self.batch_size = batch_size