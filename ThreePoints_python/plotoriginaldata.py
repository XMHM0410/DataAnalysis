import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
# 直接读取原始数据
# data = pd.read_csv('ThreePoints_Python\originaldata8.txt', header=None, delimiter='\t')
# s1 = data.iloc[:, 1].values
# s2 = data.iloc[:, 2].values
# s3 = data.iloc[:, 3].values


# # 读取滤波后的数据
df = pd.read_csv('ThreePoints_Python\Filtereddata8.csv')
s1 = df['s1'].values
s2 = df['s2'].values
s3 = df['s3'].values

alpha = 90 # 探头12夹角
beta = 90 # 探头34夹角
c1 = 1
c2 = -np.sin(np.deg2rad(alpha+beta))/np.sin(np.deg2rad(beta))
c3 = np.sin(np.deg2rad(alpha))/np.sin(np.deg2rad(beta))
x = s1*c1+s2*c2+s3*c3 # 三点法表达式

N = len(x) # 采样总点数
fs = 2000.0 # 采样频率 Hz
t_total = N/fs
t = np.arange(0,t_total,t_total/N)

# 计算频谱
fft_x = np.fft.fft(x)
freq = np.fft.fftfreq(N, d=1/fs)
amp = np.abs(fft_x)
def fft(data,fs):
    N = len(data)
    fft_x = np.fft.fft(data)
    fft_f = fft_x.copy()
    freq = np.fft.fftfreq(N, d=1/fs)
    amp = np.abs(fft_x)
    fft_f[0] = 0
    amp[0] = 0
    return freq,amp
freq_s1,amp_s1 = fft(s1,fs)
freq_s2,amp_s2 = fft(s2,fs)
freq_s3,amp_s3 = fft(s3,fs)

# 选择感兴趣的频率范围
interest_freq_range = (0.01, fs/2)  # 单位为Hz
interest_freq_mask = (freq >= interest_freq_range[0]) & (freq <= interest_freq_range[1])
freq_i = freq[interest_freq_mask]
amp_i = amp[interest_freq_mask]

# 找到极大幅值及对应的频率
max_amp_index = argrelextrema(amp_i, np.greater)
max_amp = amp_i[max_amp_index]
max_amp_freq = freq_i[max_amp_index]
# 打印所有极值点
print('Max AMP :',max_amp)
print('Max AMP Frequency:',max_amp_freq,'Hz')
# 找到最大幅值及对应频率
max_amp = np.max(amp_i)
max_amp_freq = freq_i[np.argmax(amp_i)]
# 打印最大值
print('Max AMP :',max_amp)
print('Max AMP Frequency:',max_amp_freq,'Hz')


# 绘制三点法混合信号
plt.figure(1)
plt.plot(t[0:500],x[0:500]) 
plt.xlabel('Time')
plt.ylabel('RawData1')
plt.title('Waveform')

# 混合信号频谱幅值谱
plt.figure(2)
plt.stem(freq_i, amp_i)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum')

# 绘制原始信号波形图
fig = plt.figure(3)
ax1 = fig.add_subplot(311) 
ax1.plot(t[0:500],s1[0:500])
ax2 = fig.add_subplot(312) 
ax2.plot(t[0:500],s2[0:500])
ax3 = fig.add_subplot(313) 
ax3.plot(t[0:500],s3[0:500])
# 绘制原始信号fft
fig2 = plt.figure(4)
bx1 = fig2.add_subplot(311)
bx1.stem(freq_s1[interest_freq_mask], amp_s1[interest_freq_mask])
bx2 = fig2.add_subplot(312)
bx2.stem(freq_s2[interest_freq_mask], amp_s2[interest_freq_mask])
bx3 = fig2.add_subplot(313)
bx3.stem(freq_s3[interest_freq_mask], amp_s3[interest_freq_mask])

plt.show()