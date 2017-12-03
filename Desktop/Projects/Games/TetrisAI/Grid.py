import Block
import copy
        
class Grid:
    
    def __init__(self, width, height):
        self.wallColor = (200, 200, 200)
        self.grid = []
        self.width = width
        self.height = height
        for i in range(-1, height+1):
            row = []
            for j in range(-1, width+1):
                if (j == -1 or j == width or i == -1 or i == height):
                    row.append(Block.Block(self.wallColor, False, True))
                else:
                    row.append(Block.Block((0, 0, 0), False, False))
            self.grid.append(row)
        
    def __getitem__(self, key):
        return self.grid[key]
        
    def __len__(self):
        return len(self.grid)
        
    def getRowsToDestroy(self):
        rowIndices = []
        for i in range(1, self.height + 1):
            rowIsFull = True
            for j in range(1, self.width + 1):
                if (not self.grid[i][j].isOn):
                    rowIsFull = False
                    break
            if rowIsFull:
                rowIndices.append(i)
        return rowIndices
        
    def destroyRows(self):
        indices = self.getRowsToDestroy()
        if (len(indices) == 0):
            return 0
        adjustment = 0
        for i in range(self.height):
            index = self.height - i
            if (index in indices):
                adjustment += 1
                continue
            if (adjustment == 0):
                continue
            self.grid[index + adjustment] = self.grid[index]
     
        return len(indices)
        
        
        
def states_to_grid(state, grid):
    for i in range(grid.height):
        for j in range(grid.width):
            if (state[0][i][j] == 1):
                grid[1+i][1+j].isOn = True
                grid[1+i][1+j].color = (255, 255, 255)
    piece = state[1]
    m = piece.matrix
    for i in range(piece.height):
        for j in range(piece.width):
            if m[i][j] == 1:
                #print(i+piece.topLeftYBlock, j+piece.topLeftXBlock)
                grid[i+piece.topLeftYBlock][j+piece.topLeftXBlock].isOn = True
                grid[i+piece.topLeftYBlock][j+piece.topLeftXBlock].color = piece.color
    return grid
                    
