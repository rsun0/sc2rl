import torch
import TetrisAgent
import numpy as np
import random

# This file serves as a template for Tetris AI.

class MyAgent(TetrisAgent.TetrisAgent):

    ########################
    # Constructor
    ########################
    def __init__(self, gridWidth, gridHeight, policy=None):
        TetrisAgent.TetrisAgent.__init__(self, gridWidth, gridHeight, policy)
    
    #############################
    # Define action_from_state
    #############################   
    def action_from_state(self):
        branches = self.TetrisGame.next_states()
        next_states = []
        for i in range(len(branches)):
            next_states.append(self.generate_state(branches[i]))
            
        # Determine action from policy here
        return random.randint(0, 5)
        
    
def main():
    # Define your policy
    policy = None
    # Declare Agent
    Agent = MyAgent(10, 20, policy)
    histories = []
    # Generate data for 20 random games
    for i in range(20):
        Agent.run()
        histories.append(Agent.history)
        Agent.reset()
    # Visualize random games
    for i in range(20):
        Agent.TetrisGame.visualize(histories[i])
    
    
if __name__ == "__main__":
    main()


