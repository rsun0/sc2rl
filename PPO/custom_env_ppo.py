from custom_env import DefeatRoachesEnvironment
import random

def random_agent():
    env = DefeatRoachesEnvironment(render=True)
    state, reward, done, _ = env.reset()
    print(state.shape)
    while (not done):
        action = random.randint(0, 1)
        state, reward, done, _ = env.step(action)

def main():
    random_agent()
    
if __name__ == "__main__":
    main()
