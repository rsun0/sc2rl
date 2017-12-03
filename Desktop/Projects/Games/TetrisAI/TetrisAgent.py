import torch
import Tetris
import Piece
import numpy as np
import random
import copy

#################################################################################
#
# Agent.py contains the template for an artificial intelligence agent for Tetris
#
#################################################################################


class TetrisAgent:

    # Defines the Agent
    def __init__(self, gridWidth, gridHeight, policy):
        self.gridWidth = 10
        self.gridHeight = 20
        self.pieces = Piece.loadStandardSet(gridWidth)
        self.TetrisGame = Tetris.Tetris(self.gridWidth, self.gridHeight, self.pieces)
        self.history = []
        
        self.policy = policy
        
    # Resets the environment
    def reset(self):
        self.TetrisGame.reset()
        self.history = []

    # Runs a game to completion
    def run(self):
        states = []
        action = 0
        while True:
            state = self.generate_state(self.TetrisGame)
            self.history.append(copy.deepcopy(state))
            action = self.makeActions()
            value = self.TetrisGame.step(action)
            if value == -1:
                break
        
    # Returns an array giving a binary conversion of the grid, the current piece, and the next piece
    def generate_state(self, game):
        # binary_grid is gridHeight x gridWidth numpy array, 1 if there is a block, 0 otherwise
        binary_grid = np.array([[1 if game.grid[j][i].isOn else 0 for i in range(1, self.gridWidth+1)] for j in range(1, self.gridHeight+1)])
        piece = self.TetrisGame.currentPiece
        next_piece = self.TetrisGame.nextPiece
        return [binary_grid, piece, next_piece]
        
    # Computes an action
    def makeActions(self):
        return self.action_from_state()
        
    # Returns the action space
    def actions(self):
        return range(6)
        
    ##########################################################    
    # USER DEFINED FUNCTIONS
    ##########################################################   
    
    
     
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
    Agent = TetrisAgent(10, 20, None)
    Agent.run()
    Agent.TetrisGame.visualize(Agent.history)
    #Agent.write_game_data()
        
if __name__ == "__main__":
    main()




