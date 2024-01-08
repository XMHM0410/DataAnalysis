from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 初始化卡尔曼滤波器
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([0., 0.])   # 初始状态向量
kf.F = np.array([[1., 1.], [0., 1.]])  # 状态转移矩阵
kf.H = np.array([[1., 0.]])  # 观测矩阵
kf.P *= 1000.  # 协方差矩阵初始化
kf.R = 5       # 测量噪声协方差矩阵

# 测量原始数据
data = pd.read_csv('ThreePoints_NoiseReduction\originaldata8.txt', header=None, delimiter='\t')
s1 = data.iloc[:, 1].values
s2 = data.iloc[:, 2].values
s3 = data.iloc[:, 3].values

# 使用卡尔曼滤波进行估计
filtered_s1 = []
for z1 in s1:
    kf.predict()
    kf.update(z1)
    filtered_s1.append(kf.x[0])
filtered_s2 = []
for z2 in s2:
    kf.predict()
    kf.update(z2)
    filtered_s2.append(kf.x[0])
filtered_s3 = []
for z3 in s3:
    kf.predict()
    kf.update(z3)
    filtered_s3.append(kf.x[0])

# 解决前几个数的端点问题 
front = 15 #前10个数改成前2个数的平均值
for i in range(0,front+1):
    filtered_s1[i] = np.mean(s1[i:3+i])
    filtered_s2[i] = np.mean(s2[i:3+i])
    filtered_s3[i] = np.mean(s3[i:3+i])

# 绘制结果
fig = plt.figure(1)
ax1 = fig.add_subplot(311)
ax1.plot(s1[0:100], label='Measurements')
ax1.plot(filtered_s1[0:100], label='Kalman filtered')
ax2 = fig.add_subplot(312)
ax2.plot(s2[0:100], label='Measurements')
ax2.plot(filtered_s2[0:100], label='Kalman filtered')
ax3 = fig.add_subplot(313)
ax3.plot(s3[0:100], label='Measurements')
ax3.plot(filtered_s3[0:100], label='Kalman filtered')
# ax4 = fig.add_subplot(224)
# ax4.plot(theta,Pos_sig[0:1000])
plt.show()
# 输出到文件
df = pd.DataFrame({'No': np.arange(1, 10001), 's1': filtered_s1, 's2': filtered_s2, 's3': filtered_s3})
df.to_csv('ThreePoints_NoiseReduction\Filtereddata8.csv', index=False)