import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %% 测量原始数据
data = pd.read_csv('ThreePoints_EMD_Improve\Data\9.txt', header=None, delimiter='\t')
s1 = data.iloc[:, 1].values
s2 = data.iloc[:, 2].values
s3 = data.iloc[:, 3].values
# %% 三传感器信号校零转换
def set_zero(data):
    avg = np.mean(data)
    zero_data = data - avg
    return zero_data
s1 = set_zero(s1)
s2 = set_zero(s2)
s3 = set_zero(s3)
# %%集合平均滤波，
def average_filter(data, window_size):
    filtered_data = []
    for i in range(len(data)):
        if i < window_size:
            filtered_data.append(sum(data[:i+1]) / (i+1))
        else:
            filtered_data.append(sum(data[i-window_size+1:i+1]) / window_size)
    return filtered_data
filtered_s1 = average_filter(s1,20)
filtered_s2 = average_filter(s2,20)
filtered_s3 = average_filter(s3,20)
# %%plot s1、s2、s3
plt.figure(figsize=(12,6))
plt.subplot(311)
plt.plot(s1)
plt.title('s1')
plt.subplot(312)
plt.plot(s2)
plt.title('s2')
plt.subplot(313)
plt.plot(s3)
plt.title('s3')
plt.tight_layout()
plt.figure(figsize=(12,6))
plt.subplot(311)
plt.plot(filtered_s1)
plt.title('filtered_s1')
plt.subplot(312)
plt.plot(filtered_s2)
plt.title('filtered_s2')
plt.subplot(313)
plt.plot(filtered_s3)
plt.title('filtered_s3')
plt.tight_layout()
plt.show()
# %%文件保存
df1 = pd.DataFrame({"s1":s1,"filtered_s1":filtered_s1})
df1.to_csv('ThreePoints_EMD_Improve\Data\OriginalSignalS1.csv', index=False)
df2 = pd.DataFrame({"s2":s2,"filtered_s2":filtered_s2})
df2.to_csv('ThreePoints_EMD_Improve\Data\OriginalSignalS2.csv', index=False)
df3 = pd.DataFrame({"s3":s3,"filtered_s3":filtered_s3})
df3.to_csv('ThreePoints_EMD_Improve\Data\OriginalSignalS3.csv', index=False)