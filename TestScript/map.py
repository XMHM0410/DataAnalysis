MAP = [[0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,1,1,1,0,0,0],
       [0,0,1,1,1,0,0,0],
       [0,0,1,1,1,0,0,0],
       [0,0,1,1,1,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0]]

def printMap(map):
    for row in map:
        print(row)

def getNeighbors(map, x, y):
    neighbors = []
    if x > 0:
        neighbors.append((x-1, y))
    if x < len(map[0]) - 1:
        neighbors.append((x+1, y))
    if y > 0:
        neighbors.append((x, y-1))
    if y < len(map) - 1:
        neighbors.append((x, y+1))
    return neighbors