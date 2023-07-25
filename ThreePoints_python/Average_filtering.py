from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 集合平均滤波参数
window_size = 2 # 每3个数平均一次

def average_filter(data, window_size):
    filtered_data = []
    for i in range(len(data)):
        if i < window_size:
            filtered_data.append(sum(data[:i+1]) / (i+1))
        else:
            filtered_data.append(sum(data[i-window_size+1:i+1]) / window_size)
    return filtered_data
# 测量原始数据
data = pd.read_csv('ThreePoints_Python\originaldata8.txt', header=None, delimiter='\t')
s1 = data.iloc[:, 1].values
s2 = data.iloc[:, 2].values
s3 = data.iloc[:, 3].values

# 三传感器信号校零转换
def set_zero(data):
    avg = np.mean(data)
    zero_data = data - avg
    return zero_data
s1 = set_zero(s1)
s2 = set_zero(s2)
s3 = set_zero(s3)

# 使用集合平均滤波进行估计
filtered_s1 = average_filter(s1,window_size)
filtered_s2 = average_filter(s2,window_size)
filtered_s3 = average_filter(s3,window_size)

# 绘制结果
fig = plt.figure(1)
ax1 = fig.add_subplot(311)
ax1.plot(s1[0:1000], label='Measurements')
ax1.plot(filtered_s1[0:1000], label='Kalman filtered')
ax2 = fig.add_subplot(312)
ax2.plot(s2[0:1000], label='Measurements')
ax2.plot(filtered_s2[0:1000], label='Kalman filtered')
ax3 = fig.add_subplot(313)
ax3.plot(s3[0:1000], label='Measurements')
ax3.plot(filtered_s3[0:1000], label='Kalman filtered')
plt.show()
# 输出到文件
df = pd.DataFrame({'No': np.arange(1, 10001), 's1': filtered_s1, 's2': filtered_s2, 's3': filtered_s3})
df.to_csv('ThreePoints_Python\Filtereddata8.csv', index=False)