from custom_env import DefeatRoachesEnvironment
import random

def random_agent():
    env = DefeatRoachesEnvironment()
    state = env.reset()
    print(state.shape)
    while (not state[2]):
        action = random.randint(0, 1)
        state = env.step(action)

def main():
    random_agent()
