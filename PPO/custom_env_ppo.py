from custom_env import MinigameEnvironment
from modified_state_space import state_modifier
import random
import time

def random_agent():
    env = MinigameEnvironment(render=True,step_multiplier=4,state_modifier_func=state_modifier.modified_state_space)
    state, reward, done, _ = env.reset()
    print(state.shape)
    for i in range(10): 
        print("Beginning game %d" % (i+1))
        t = 0
        while (not done):
            if (t % 2 == 0):
                
                topleft = [random.randint(0,4), random.randint(0,4)]
                botright = [random.randint(5,9), random.randint(5,9)]
                
                state, reward, done, _ = env.step(0, topleft=topleft, botright=botright)
            else:
                action = random.randint(0, 10)
                state, reward, done, _ = env.step(action)
            t += 1
        state, reward, done, _ = env.reset()

def main():
    random_agent()
    
if __name__ == "__main__":
    main()
