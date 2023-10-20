from map import MAP ,printMap,getNeighbors
from move import Move,Judge
# %%设置起点和终点
start = [1, 1]
end = [6,6]
map = MAP.copy()
map[start[0]][start[1]] = 3 # 设3为起点终点和路径的标识
map[end[0]][end[1]] = 3
# printMap(MAP)
# %%走路并记录点位信息
is_end = False
current = start.copy()
step = 0
path = []
while not is_end:
    map[current[0]][current[1]] = 2 # 记录当前点位信息
    step += 1
    is_end = Judge.isEnd(current,end) # 判断当前点是否到达终点
    pos = Judge.position(current,end)# 判断终点在当前点的哪个方向
    print(pos)
    # 判断当前点周围8个点的障碍物情况
    current_try = current.copy()
    # 向没有障碍物的与目标方向相近的方向移动一格
    Move.moveDownRight(current)
# %%打印出地图和总步数
printMap(map)
print("总步数：",step)