class Move():
    def moveUp(current):
        current[0] -=1
        return current
    def moveDown(current):
        current[0] +=1
        return current
    def moveLeft(current):
        current[1] -=1
        return current
    def moveRight(current):
        current[1] +=1
        return current
    def moveUpLeft(current):
        current[0] -=1
        current[1] -=1
        return current
    def moveUpRight(current):
        current[0] -=1
        current[1] +=1
        return current
    def moveDownLeft(current):
        current[0] +=1
        current[1] -=1
        return current
    def moveDownRight(current):
        current[0] +=1
        current[1] +=1
        return current
class Judge():
    def isEnd(current,end):
        if current == end:
            return True
        else:
            return False
    def isOne(map,current):
        if map[current[0]][current[1]] == 1:
            return True
        else:
            return False
    def position(current,end):
        if (current[0] - end[0] >0) & (current[1] - end[1] == 0):
            return "u"
        if (current[0] - end[0] <0) & (current[1] - end[1] == 0):
            return "d"
        if (current[0] - end[0] == 0) & (current[1] - end[1] >0):
            return "l"
        if (current[0] - end[0] == 0) & (current[1] - end[1] <0):
            return "r"
        if (current[0] - end[0] >0) & (current[1] - end[1] > 0):
            return "ul"
        if (current[0] - end[0] <0) & (current[1] - end[1] > 0):
            return "ur"
        if (current[0] - end[0] >0) & (current[1] - end[1] < 0):
            return "dl"
        if (current[0] - end[0] <0) & (current[1] - end[1] < 0):
            return "dr"
        if (current[0] - end[0] == 0) & (current[1] - end[1] == 0):
            return "end"
