import numpy as np
import matplotlib.pyplot as plt

# 定义固定点和测量点
fixed_points = [(0, 0), (0, 1)]  # 两个固定点
measuring_points = [(0.5, 0), (0.5, 1)]  # 两个测量点

# 定义主轴旋转角度
angles = np.linspace(0, 2*np.pi, 360)

# 初始化距离差异列表
distance_differences = []

# 测量每个角度下的距离差异
for angle in angles:
    # 旋转主轴
    rotated_measuring_points = []
    for point in measuring_points:
        # 将测量点旋转
        x = (point[0] - fixed_points[0][0]) * np.cos(angle) - (point[1] - fixed_points[0][1]) * np.sin(angle) + fixed_points[0][0]
        y = (point[0] - fixed_points[0][0]) * np.sin(angle) + (point[1] - fixed_points[0][1]) * np.cos(angle) + fixed_points[0][1]
        rotated_measuring_points.append((x, y))

    # 计算距离差异
    distance_difference = np.sqrt((rotated_measuring_points[0][0] - rotated_measuring_points[1][0])**2 + (rotated_measuring_points[0][1] - rotated_measuring_points[1][1])**2)
    distance_differences.append(distance_difference)

# 将角度转换为度数
angles_degrees = angles * 180 / np.pi

# 绘制极坐标图
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.plot(angles, distance_differences)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.set_title('Spindle Runout Error')
plt.show()
