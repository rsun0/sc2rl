"""

Created by Michael McGuire

The logic here is to create a generalized interface for an arbitrary number of agents to train.
To create a new agent for a new map or purpose, we want to only have to define a concise set of new functions.
These functions are passed into the train template.

"""



"""
    Takes in an arbitrary experiment and runs a train loop through Starcraft 2.
    @param experiment: variable of type Experiment. Contains train_step, sample, forward, and criterion functions, as well as an optimizer and many various hyperparameter variables, for up to two agents.

"""


def train_experiment(experiment):

    agents = experiment.agents
    train_length = max([agents[i].train_length for i in range(len(agents))])
    
    env = experiment.env
    
    env_obs = env.reset()
    
    for frame in range(train_length):
        
        actions = []
        for i in range(len(agents))
            
            agent = agents[i]
            
            if frame % agent.train_every == 0:
                agent.train()
            
            agent_state = env_obs[i].observation
            agent_action, agent_modified_state = agent.sample(agent_state)
            converted_action = agent.action_space_converter(agent_action)
            
            agent_states[i] = agent_modified_state
            agent_actions[i] = converted_action
            
        env_obs = env.step(agent_actions)
        agent_rewards = [env_obs[i].reward for i in range(len(env_obs))]
        agent_dones = [env_obs[i].last() for i in range(len(env_obs))]
        
        if (frame != 0):
            for i in range(len(agents)):
                ### Push memory
                agent_modified_state = agent_states[i]
                agent_action = agent_actions[i]
                agent_reward = agent_rewards[i]
                agent_done = agent_dones[i]
                agent.memory.push([agent_modified_space, agent_action, agent_reward, agent_done])
                
        if (np.array(agent_dones) == False).all():
            env_obs = env.reset()
            agent_states = [None for i in range(len(agents))]
            agent_actions = [None for i in range(len(agents))]
            agent_rewards = [0 for i in range(len(agents))]
            agent_dones = [False for i in range(len(agents))]
            
        
            
            
            
    
    
   
    
