import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
# %%定义正弦波的参数
amplitudes =  [1.0,0.8,0.6,0.4,0.2,0.1,0.1,0.1,0.1,0.08,0.06,0.04 ,0.02,0.01]  # 振幅
frequencies = [50 ,100,120,200,250,300,360,400,500,600 ,700 ,800 ,900 ,1000]  # 频率
sampling_rate = 2000  # 采样率
duration = 1  # 信号时长
# %%生成时间序列
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
# %%生成正弦波信号
signals = []
for amplitude, frequency in zip(amplitudes, frequencies):
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    signals.append(signal)
# %%合并信号
mixed_signal = np.sum(signals, axis=0)
# %%添加高斯白噪音
# noise = np.random.normal(0, 0.1, len(t))
# mixed_signal = mixed_signal + noise
# %%创建DataFrame并保存为.csv文件
# df = pd.DataFrame({'Time': t, 'Mixed Signal': mixed_signal})
# df.to_csv('mixed_signal.csv', index=False)
# %%仿真信号波形图像
plt.plot(t[0:201], mixed_signal[0:201])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Mixed Signal')
# %%仿真信号fft
N = len(t) # 采样总点数
fft_ms = np.fft.fft(mixed_signal)
freq = np.fft.fftfreq(N, d=1/sampling_rate)
amp = np.abs(fft_ms)/(N/2)
# %%选择感兴趣的频率范围
interest_freq_range = (0.01, sampling_rate/2)  # 单位为Hz
interest_freq_mask = (freq >= interest_freq_range[0]) & (freq <= interest_freq_range[1])
freq_i = freq[interest_freq_mask]
amp_i = amp[interest_freq_mask]
# %%找到极大幅值及对应的频率
max_amp_index = argrelextrema(amp_i, np.greater)
max_amp = amp_i[max_amp_index]
max_amp_freq = freq_i[max_amp_index]
# %%打印所有极值点
print('Max AMP :',max_amp)
print('Max AMP Frequency:',max_amp_freq,'Hz')
# %%混合信号频谱幅值谱
plt.figure(2)
plt.stem(freq_i, amp_i)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum')
plt.show()