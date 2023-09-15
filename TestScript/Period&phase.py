import numpy as np
import matplotlib.pyplot as plt

# 生成正弦波
t = np.arange(0, 1, 1/2000)  # 时间序列
f = 50  # 正弦波频率
A = 1  # 正弦波幅值
phi = np.pi/4  # 正弦波相位
x = A * np.sin(2*np.pi*f*t + phi)  # 正弦波信号

# 从文件导入波形

# 计算周期和相位
dt = np.diff(t)  # 采样点之间的时间间隔
T = 2 * np.mean(dt)  # 波形周期
phi = np.arcsin(x[0]/A) - 2*np.pi*f*t[0]  # 波形相位

# 反推回转轴转速
e = 0.1  # 偏心距
omega = 4*np.pi*e/T  # 回转轴转速

# 转换为r/min单位
rpm = omega * 60 / (2*np.pi)

# 打印结果
print("波形周期为：", T)
print("波形相位为：", phi)
print("回转轴转速为：", rpm, "r/min")

# 输出极坐标图
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.plot(2*np.pi*t, x)
ax.set_rmax(1)
ax.set_rticks([0.5, 1])  # less radial ticks
ax.set_rlabel_position(-22.5)  # move radial labels away from plotted line
ax.grid(True)

plt.show()