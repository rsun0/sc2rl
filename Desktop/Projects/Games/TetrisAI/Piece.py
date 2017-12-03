import numpy as np

class Piece:

    def __init__(self, npMatrix, windowWidth, color):
        self.matrix = npMatrix
        self.windowWidth = windowWidth
        self.width = npMatrix.shape[1]
        self.height = npMatrix.shape[0]
        
        #Block coords
        self.topLeftXBlock = int((windowWidth + 2 - self.width) / 2)
        self.topLeftYBlock = 1
        
        self.color = color
        
    def movePiece(self, dx, dy):
        self.topLeftXBlock += dx
        self.topLeftYBlock += dy
        self.update()
        
    def update(self):
        self.width = self.matrix.shape[1]
        self.height = self.matrix.shape[0]
        
    def reset(self):
        self.topLeftXBlock = int((self.windowWidth + 2 - self.width) / 2)
        self.topLeftYBlock = 1
        
    #ACTIONS
        
    def rotateCW(self):
        self.matrix = np.rot90(self.matrix, 3)
        self.update()
        
    def rotateCoCW(self):
        self.matrix = np.rot90(self.matrix)
        self.update()
        
    def moveRight(self):
        self.movePiece(1, 0)
    
    def moveLeft(self):
        self.movePiece(-1, 0)
        
        
        
#STATIC METHODS
def loadStandardSet(gridWidth):
    p1 = Piece(np.array([[1, 0, 0], [1, 1, 1]]), gridWidth, (255, 0, 0))
    p2 = Piece(np.array([[0, 0, 1], [1, 1, 1]]), gridWidth, (0, 0, 255))
    p3 = Piece(np.array([[1, 1], [1, 1]]), gridWidth, (255, 255, 0))
    p4 = Piece(np.array([[1],[1],[1],[1]]), gridWidth, (255, 200, 200))
    p5 = Piece(np.array([[0, 1, 0], [1, 1, 1]]), gridWidth, (180, 180, 180))
    p6 = Piece(np.array([[0, 1, 1], [1, 1, 0]]), gridWidth, (0, 255, 0))
    p7 = Piece(np.array([[1, 0], [1, 1], [0, 1]]), gridWidth, (255, 0, 255))
    return [p1, p2, p3, p4, p5, p6, p7]
