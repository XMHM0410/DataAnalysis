import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%读取滤波后的数据
df = pd.read_csv('ThreePoints_Python\Data\Filtereddata8.csv')
s1 = df['filtered_s1'].values
s2 = df['filtered_s2'].values
s3 = df['filtered_s3'].values
# %%定义基本参数
rpm = 6000 # 转速
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
# %%绘制原始信号波形图
plt.figure(1)
plt.plot(t[0:1000],x[0:1000]) # 前1000条
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Waveform')
# %%计算频谱
fft_x = np.fft.fft(x)
freq = np.fft.fftfreq(N, d=1/fs)
amp = np.abs(fft_x)
# %%选择感兴趣的频率范围
interest_freq_range = (0.01, 1000.0)  # 单位为Hz
interest_freq_mask = (freq >= interest_freq_range[0]) & (freq <= interest_freq_range[1])
freq_i = freq[interest_freq_mask]
amp_i = amp[interest_freq_mask]
# %%找到最大幅值及对应的频率
max_amp = np.max(amp_i)
max_amp_freq = freq_i[np.argmax(amp_i)]
# %%绘制原始信号频谱幅值谱
plt.figure(2)
# plt.stem(freq[:len(freq)//2], amp[:len(amp)//2])
plt.stem(freq_i, amp_i)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum')
# plt.xlim(interest_freq_range)
#'freq_i':freq_i,'amp_i':amp_i,
plt.show()
# %%输出到文件
df = pd.DataFrame({'x': x})
df.to_csv('ThreePoints_Python\Data\ThreePointsResultData.csv', index=False)
df = pd.DataFrame({'freq': freq, 'amp': amp})
df.to_csv('ThreePoints_Python\Data\ThreePointsResultFreqAmp.csv', index=False)