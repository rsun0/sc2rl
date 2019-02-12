import custom_env
import random
from modified_state_space import state_modifier










def main():

    env = custom_env.MinigameEnvironment(state_modifier.modified_state_space,
                                            map_name_="DefeatRoaches",
                                            render=True,
                                            step_multiplier=8)
                                            
    
    topleft = [10, 10]
    botright = [20, 20]
    
    
    state, reward, done, _ = env.reset()
    
    while True:
        topleft = [random.randint(0,83), random.randint(0,83)]
        botright = [random.randint(0,83), random.randint(0,83)]
        action = random.randint(0, 4)
        state, reward, done, _ = env.step(action, topleft=topleft, botright=botright)
        if done:
            state, reward, done, _ = env.reset()    
    
if __name__ == "__main__":
    main()
