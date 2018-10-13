from custom_env import DefeatRoachesEnvironment
import random

def random_agent():
    env = DefeatRoachesEnvironment(render=True,step_multiplier=4)
    state, reward, done, _ = env.reset()
    print(state.shape)
    for i in range(10): 
        while (not done):
            action = random.randint(0, 1)
            state, reward, done, _ = env.step(action)
        state, reward, done, _ = env.reset()

def main():
    random_agent()
    
if __name__ == "__main__":
    main()
