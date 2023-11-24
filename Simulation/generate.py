import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
# %%定义正弦波的参数
amplitudes =  [1.0,0.8,0.6,0.4,0.2,0.1,0.1,0.1,0.1,0.08,0.06,0.04 ,0.02,0.01]  # 振幅
frequencies = [50 ,100,120,200,250,300,360,400,500,600 ,700 ,800 ,900 ,1000]  # 频率
sampling_rate = 2000  # 采样率
duration = 5  # 信号时长
# %%生成时间序列和角度序列
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
# deg = np.arange(0,360,0.18)
# %%生成正弦波信号
signals = []
for amplitude, frequency in zip(amplitudes, frequencies):
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    signals.append(signal)
# %%合并信号
mixed_signal = np.sum(signals, axis=0)
# %%添加高斯白噪音
noise = np.random.normal(0, 0.1, len(t))
mixed_signal = mixed_signal + noise
# %%创建DataFrame并保存为.csv文件
# %%保存文件
df3 = pd.DataFrame({
    "t":t,
    # "deg":deg,
    "mixed_signal":mixed_signal
})
df3.to_csv('Simulation\Data\simSignal.csv',index=False)
# %%仿真信号波形图像
plt.plot(t[0:1000], mixed_signal[0:1000])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Mixed Signal')
plt.show()