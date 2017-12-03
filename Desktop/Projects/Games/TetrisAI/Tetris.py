
import pygame
import random
import Block
import Piece
import Grid
import numpy as np
import copy

class Tetris:

    def __init__(self, gridWidth, gridHeight, pieces):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.pieces = pieces
        
        self.grid = Grid.Grid(self.gridWidth, self.gridHeight)
        self.score = 0
        self.currentPiece = pieces[random.randint(0, len(pieces))-1]
        self.nextPiece = pieces[random.randint(0, len(pieces))-1]
        self.rowValue = 100
        
    def reset(self):
        self.grid = Grid.Grid(self.gridWidth, self.gridHeight)
        self.score = 0
        self.currentPiece = self.pieces[random.randint(0, len(self.pieces))-1]
        self.nextPiece = self.pieces[random.randint(0, len(self.pieces))-1]
        
    def step(self, action):
        if (not self.canMoveDown()) and self.currentPiece.topLeftYBlock == 1:
            return -1
        if (action == 0):
            self.moveLeft()
        if (action == 1):
            self.moveRight()
        if (action == 2):
            self.rotateCW()
        if (action == 3):
            self.rotateCoCW()
        if (action == 4):
            self.drop()
        if (action == 5):
            pass
        self.down()
        self.destroyRows()
        return 0
            
    def next_states(self):
        states = []
        for i in range(6):
            s = copy.deepcopy(self)
            s.step(i)
            states.append(s)
        return states
        
    def updatePieces(self):
        self.nextPiece.reset()
        self.currentPiece = self.nextPiece
        self.nextPiece = self.pieces[random.randint(0, len(self.pieces)) - 1]
        
        
    def addPieceToGrid(self):
        m = self.currentPiece.matrix
        p = self.currentPiece
        for i in range(p.width):
            for j in range(p.height):
                if m[j][i] == 1:
                    self.grid[p.topLeftYBlock + j][p.topLeftXBlock + i] = Block.Block(self.currentPiece.color, True, False)
        self.updatePieces()
        
    def drawGrid(self, surface, grid, topLeftX, topLeftY, boxWidth, margin, piece):
        #draws grid
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                pygame.draw.rect(surface, grid[i][j].color, (topLeftX + boxWidth * j, topLeftY + boxWidth * i, boxWidth - margin, boxWidth - margin))
                
                
        #draws currentPiece
        """
        matrix = piece.matrix
        x = piece.topLeftXBlock
        y = piece.topLeftYBlock
        for i in range(piece.width):
            for j in range(piece.height):
                if (piece.matrix[j][i] == 1):
                    pygame.draw.rect(surface, piece.color, (topLeftX + boxWidth * (x + i), topLeftY + boxWidth * (y + j), boxWidth - margin, boxWidth - margin))
        """
        
        
    def destroyRows(self):
        num_rows = self.grid.destroyRows()
        self.score += self.rowValue * num_rows
        
    def movePieceDown(self):
        self.movePiece(0, 1)
        
    #Moves the piece dx in the x direction and dy in the y direction
    def movePiece(self, dx, dy):
        self.currentPiece.movePiece(dx, dy)
        width = self.currentPiece.width
        height = self.currentPiece.height
        widthRange = []
        heightRange = []
        if dx < 0:
            #iterate through x values left to right
            widthRange = range(width)
        elif dx > 0:
            #iterate through x values right to left
            widthRange = range(width-1, 0, -1)
            
        if dy < 0:
            #iterate through y values up to down
            heightRange = range(height)
        elif dy > 0:
            #iterate through y values down to up
            heightRange = range(height-1, 0, -1)
            
        for i in widthRange:
            for j in heightRange:
                self.grid[j][i] = self.grid[j-dy][i-dx]
            
    def canMoveDown(self):
        x = self.currentPiece.topLeftXBlock
        y = self.currentPiece.topLeftYBlock
        h = self.currentPiece.height
        w = self.currentPiece.width
        bottomRow = y + h - 1
        if bottomRow >= self.gridHeight:
            return False
        for i in range(w):
            for j in range(h):
                if (self.currentPiece.matrix[j][i] == 1 and self.grid[y+j+1][x+i].isOn):
                    return False
        return True
            
    def canMoveLeft(self):
        if (self.currentPiece.topLeftXBlock <= 1):
            return False
        for i in range(self.currentPiece.height):
            if (self.grid[self.currentPiece.topLeftYBlock + i][self.currentPiece.topLeftXBlock - 1].isOn):
                return False
        return True
        
    def canMoveRight(self):
        if (self.currentPiece.topLeftXBlock + self.currentPiece.width - 1 >= self.gridWidth):
            return False
        for i in range(self.currentPiece.height):            
            if (self.grid[self.currentPiece.topLeftYBlock + i][self.currentPiece.topLeftXBlock + self.currentPiece.width].isOn and self.grid[self.currentPiece.topLeftYBlock + i][self.currentPiece.topLeftXBlock + self.currentPiece.width - 1].isOn):
                return False
        return True
        
    def drop(self):
        while (self.canMoveDown()):
            self.movePieceDown()
        self.addPieceToGrid()
        
    def valid(self):
        tlx = self.currentPiece.topLeftXBlock
        tly = self.currentPiece.topLeftYBlock
        h = self.currentPiece.height
        w = self.currentPiece.width
        m = self.currentPiece.matrix
        for i in range(h):
            for j in range(w):
                if (self.grid[tly+i][tlx+j].isOn and m[i][j] == 1):
                    return False
                if (tly+i < 1 or tly+i >self.gridHeight or tlx+j < 1 or tlx+j > self.gridWidth):
                    return False
        return True
                     
    def moveLeft(self):
        self.currentPiece.moveLeft()  
        if (not self.valid()):
            self.currentPiece.moveRight()
        
    def moveRight(self):
        self.currentPiece.moveRight()  
        if (not self.valid()):
            self.currentPiece.moveLeft()
    def rotateCoCW(self):
        self.currentPiece.rotateCoCW()
        if (not self.valid()):
            self.currentPiece.rotateCW()
        
    def rotateCW(self):
        self.currentPiece.rotateCW()
        if (not self.valid()):
            self.currentPiece.rotateCoCW()
        
    def down(self):
        self.movePieceDown()
        if (not self.valid()):
            self.movePiece(0, -1)
            self.addPieceToGrid()
            
            
            
    def visualize(self, states):
        surface = pygame.display.set_mode((600, 600))
        clock = pygame.time.Clock()
        for i in range(len(states)):
            #print (states[i][1].topLeftXBlock, states[i][1].topLeftYBlock)
            grid = Grid.Grid(self.gridWidth, self.gridHeight)
            grid = Grid.states_to_grid(states[i], grid)
            self.drawGrid(surface, grid, 100, 100, 20, 1, states[i][1])
            pygame.display.update()
            
            
            
            clock.tick(10)
        
