from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# 频域滤波参数
threshold = 0.25 # 频域滤波幅值阈值/但是三个可能不一样
fs = 2000.0

def fftfiltering(data,fs,threshold):
    N = len(data)
    fft_x = np.fft.fft(data)
    fft_f = fft_x.copy()
    freq = np.fft.fftfreq(N, d=1/fs)
    amp = np.abs(fft_x)
    for i in range(len(freq)):
        if amp[i] < threshold:
            fft_f[i] = 0
    fft_f[0] = 0
    amp[0] = 0
    amp_f = np.abs(fft_f)
    ifft_f = np.fft.ifft(fft_f)
    ifft_data = ifft_f.real
    return ifft_data,amp_f,freq,amp

# 使用卡尔曼滤波进行估计
filtered_s1,fft_s1,freq_s1,amp_s1 = fftfiltering(s1,fs,threshold)
filtered_s2,fft_s2,freq_s2,amp_s2 = fftfiltering(s2,fs,threshold)
filtered_s3,fft_s3,freq_s3,amp_s3 = fftfiltering(s3,fs,threshold)

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

fig2 = plt.figure(2)
bx1 = fig2.add_subplot(311)
bx1.stem(freq_s1, amp_s1)
bx2 = fig2.add_subplot(312)
bx2.stem(freq_s2, amp_s2)
bx3 = fig2.add_subplot(313)
bx3.stem(freq_s3, amp_s3)

fig3 = plt.figure(3)
cx1 = fig3.add_subplot(311)
cx1.stem(freq_s1, fft_s1)
cx2 = fig3.add_subplot(312)
cx2.stem(freq_s2, fft_s2)
cx3 = fig3.add_subplot(313)
cx3.stem(freq_s3, fft_s3)

plt.show()
# 输出到文件
df = pd.DataFrame({'No': np.arange(1, 10001), 's1': filtered_s1, 's2': filtered_s2, 's3': filtered_s3})
df.to_csv('ThreePoints_Python\Filtereddata8.csv', index=False)