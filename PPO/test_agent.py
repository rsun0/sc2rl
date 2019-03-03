import custom_env
import random
from modified_state_space import state_modifier
import argparse
from models import *
import torch
import time

# Used to debug memory sizes
from pympler import asizeof

def default_test():

    env = custom_env.MinigameEnvironment(state_modifier.graph_conv_modifier,
                                            map_name_="DefeatRoaches",
                                            render=True,
                                            step_multiplier=1)
                                            
    topleft = [10, 10]
    botright = [20, 20]
    
    
    state, reward, done, _ = env.reset()
    
    while True:
        topleft = None
        #topleft = [random.randint(0,83), random.randint(0,83)]
        #botright = [random.randint(0,83), random.randint(0,83)]
        action_ind = random.randint(0,len(state[2].nonzero()[0])-1)
        action = state[2].nonzero()[0][action_ind]
        if (action == 1 or action == 2):
            topleft = [random.randint(0,83), random.randint(0,83)]
            state, reward, done, _ = env.step(action, topleft=topleft) #, topleft=topleft, botright=botright)
        else:
            state, reward, done, _ = env.step(action)
        if done:
            state, reward, done, _ = env.reset()    

def GraphConvTest():
    
    env = custom_env.MinigameEnvironment(state_modifier.graph_conv_modifier,
                                            map_name_="DefeatRoaches",
                                            render=True,
                                            step_multiplier=1)

    nonspatial_act_size, spatial_act_depth = env.action_space
                                            
    topleft = [10, 10]
    botright = [20, 20]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent = GraphConvNet(nonspatial_act_size, spatial_act_depth, device).to(device)
    
    state, reward, done, _ = env.reset()
    
    while True:
        G, X, avail_actions = state
        G = np.expand_dims(G, 0)
        X = np.expand_dims(X, 0)
        avail_actions = np.expand_dims(avail_actions, 0)
        
        agent(G, X, avail_actions)
        
        topleft = None
        #topleft = [random.randint(0,83), random.randint(0,83)]
        #botright = [random.randint(0,83), random.randint(0,83)]
        action_ind = random.randint(0,len(state[2].nonzero()[0])-1)
        action = state[2].nonzero()[0][action_ind]
        if (action == 1 or action == 2):
            topleft = [random.randint(0,83), random.randint(0,83)]
            state, reward, done, _ = env.step(action, topleft=topleft) #, topleft=topleft, botright=botright)
        else:
            state, reward, done, _ = env.step(action)
        if done:
            state, reward, done, _ = env.reset()  
            
            
            

def DeepMind2017Test():
    
    env = custom_env.MinigameEnvironment(state_modifier.modified_state_space,
                                            map_name_="DefeatRoaches",
                                            render=True,
                                            step_multiplier=8)
                                 
    nonspatial_act_size, spatial_act_depth = env.action_space
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent = DeepMind2017Net(nonspatial_act_size, spatial_act_depth, device).to(device)
    state, reward, done, _ = env.reset()
    screen, minimap, nonspatial_in, avail_actions = state
    print("loop beginning")
    while True:
        
        t1 = time.time()
        spatial_pol, nonspatial_pol, value, action = agent(screen, minimap, nonspatial_in, avail_actions, choosing=True)
        spatial_action, nonspatial_action = action
        #print("Action time: %f" % (time.time() - t1))
        
        t1 = time.time()
        state, reward, done, _ = env.step(nonspatial_action, spatial_action[0], spatial_action[1])
        #print("Env time: %f" % (time.time() - t1))
        if done:
            state, reward, done, _ = env.reset()
            
        screen, minimap, nonspatial_in, avail_actions = state
        
        print(asizeof.asizeof(state))
        
        
     
                                            
    



def main():

    method = 'graphconv'    
    
    if (method == 'default'):
        default_test()
    elif (method == 'deepmind2017'):
        DeepMind2017Test()
    elif method == 'graphconv':
        GraphConvTest()
    
if __name__ == "__main__":
    main()
