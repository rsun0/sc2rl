"""

Created by Michael McGuire and Ray Sun

Purpose: a generalized interface that can train arbitrary models and configurations.
To create a new agent for a new map or purpose,
we want to only have to define a concise set of new functions.
These functions and settings are passed into the Experiment constructor.

"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import deque   

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
        averages = [0 for i in range(len(self.agents))]
        evaluation_rewards = [deque(maxlen=100) for i in range(len(self.agents))]
        averages = [[] for i in range(len(self.agents))]
        

        for e in range(self.run_settings.num_episodes):
            # Initialize episode
            env_states, rewards, dones, metainfo = self.custom_env.reset()
            done = dones[0]
            
            # Initialize scores to starting reward (probably 0)
            scores = rewards
            step = 0
            
            while not done:
               
                states = [self.agents[a].state_space_converter(env_states[a]) for a in range(len(self.agents))]
            
                total_frame_count += 1

                # Train agents
                if total_frame_count % self.run_settings.train_every == 0:
                    for agent in self.agents:
                        agent.train(self.run_settings)
                        
                # Save agent model
                if total_frame_count % self.run_settings.save_every == 0:
                    for agent in self.agents:
                        agent.save()

                # Get actions
                actions = [self.agents[a].sample(states[a]) for a in range(len(self.agents))]
                env_actions = [self.agents[a].action_space_converter(actions[a]) for a in range(len(self.agents))]
                # Take environment step
                env_states, rewards, dones, metainfo = self.custom_env.step(env_actions)
                
                
                # Update scores
                scores = [scores[a] + rewards[a] for a in range(len(self.agents))]
                # Push to agent Memories
                for a in range(len(self.agents)):
                    self.agents[a].push_memory(states[a], actions[a], rewards[a], dones[a])
                
                done = dones[0]
                step += 1
                
                if done:
                    curr_averages = []
                    for i in range(len(evaluation_rewards)):
                        evaluation_rewards[i].append(scores[i])
                        curr_averages.append(np.mean(evaluation_rewards[i]))
                        averages[i].append(curr_averages[i])
                        
                    if len(scores) == 1:
                        scores = scores[0]
                        curr_averages = curr_averages[0]
                    recent_mean = np.mean(evaluation_rewards)
                    print("Game %d ended after %d steps. Game score: %s. Averages: %s" % (e+1, step, scores, curr_averages))
                    
            if (e % 50 == 0 and e != 0):
                self.plot_results(averages)
                
    def plot_results(self, averages):
        plt.figure(1)
        plt.clf()
        plt.suptitle("Training results")
        plt.title("Ray Sun, David Long, Michael McGuire")
        plt.xlabel("Training iteration")
        plt.ylabel("Average score")
        for i in range(len(averages)):
            plt.plot(averages[i])
        plt.pause(0.005)
            
class CustomEnvironment():
    def step(self, actions):
        """
        :actions: A list of actions, one for each agent

        :returns: states, rewards, done, metainfo
        states is an list of states, one for each agent
        rewards is an list of rewards, one for each agent
        done is the flag for the terminal frame
        metainfo is auxiliary data used by the controlling system (not visible to agents)
        """
        raise NotImplementedError()

    def reset(self):
        """
        Should work even if reset is called multiple times in a row.

        :returns: states, rewards, done, metainfo
        states is an list of states, one for each agent
        rewards is an list of rewards, one for each agent
        done is the flag for the terminal frame
        metainfo is auxiliary data used by the controlling system (not visible to agents)
        """
        raise NotImplementedError()


class RunSettings:
    def __init__(self, num_episodes, num_epochs, batch_size, train_every, save_every):
        """
        :param num_episodes: The total number of episodes to play
        :param num_epochs: The number of update iterations for each experience set
        :param batch_size: The number of experiences to process at once
        :param train_every: Update the networks every X frames
        :param save_every: Save the model every X frames
        """
        self.num_episodes = num_episodes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_every = train_every
        self.save_every = save_every
