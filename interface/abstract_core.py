"""

Created by Michael McGuire and Ray Sun

Purpose: a generalized interface that can train arbitrary models and configurations.
To create a new agent for a new map or purpose,
we want to only have to define a concise set of new functions.
These functions and settings are passed into the Experiment constructor.

"""

import numpy as np

class Experiment:
    def __init__(self, agents, custom_env, run_settings):
        self.agents = agents
        self.custom_env = custom_env
        self.run_settings = run_settings

    def train(self):
        """
        Trains the agent(s) using the custom environment
        """
        total_frame_count = 0

        for e in range(self.run_settings.num_episodes):
            # Initialize episode
            states, rewards, done, info = self.custom_env.reset()
            # Initialize scores to starting reward (probably 0)
            scores = rewards
            
            while not done:
                
                states = [self.agents[a].state_space_converter(states[a]) for a in range(len(self.agents))]
            
                total_frame_count += 1

                # Train agents
                if total_frame_count % self.run_settings.train_every == 0:
                    for agent in self.agents:
                        agent.train()
                        
                        # Save agent model
                        agent.save()

                # Get actions
                actions = [self.agents[a].sample(states[a]) for a in range(len(self.agents))]
                env_actions = [self.agents[a].action_space_converter(actions[a]) for a in range(len(self.agents))]
                # Take environment step
                next_states, rewards, done, info = self.custom_env.step(env_actions)
                
                # Update scores
                scores = [scores[a] + rewards[a] for a in range(len(self.agents))]
                # Push to agent Memories
                for a in range(len(self.agents)):
                    self.agents[a].memory.push(states[a], actions[a], rewards[a], done)

                # TODO Save models and record metrics
                states = next_states
            

class CustomEnvironment():
    def step(self, actions):
        """
        :actions: A list of actions, one for each agent

        :returns: states, rewards, done, info
        states is an list of states, one for each agent
        rewards is an list of rewards, one for each agent
        done is the flag for the terminal frame
        info is auxiliary data used by the controlling system (not visible to agents)
        """
        raise NotImplementedError()

    def reset(self):
        """
        Should work even if reset is called multiple times in a row.

        :returns: states, rewards, done, info
        states is an list of states, one for each agent
        rewards is an list of rewards, one for each agent
        done is the flag for the terminal frame
        info is auxiliary data used by the controlling system (not visible to agents)
        """
        raise NotImplementedError()


class RunSettings:
    def __init__(self, num_episodes, num_epochs, batch_size, train_every):
        """
        :param num_episodes: The total number of episodes to play
        :param num_epochs: The number of update iterations for each experience set
        :param batch_size: The number of experiences to process at once
        :param train_every: Update the networks every X frames
        """
        self.num_episodes = num_episodes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_every = train_every
