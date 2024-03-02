import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
# %%定义正弦波的参数
amplitudes =  [0.01,0.01,0.01,0.10,0.01,0.01,0.01,1.00,0.01,0.01,0.01,0.10,0.01,0.01,0.01,0.50,0.01,0.01,0.01,0.10,0.01,0.01,0.01,0.09,0.01,0.01,0.01,0.08,0.01,0.01,0.01,0.07,0.01,0.01,0.01,0.06,0.01,0.01,0.01,0.05]  # 振幅
frequencies = [25  ,50  ,75  ,100 ,125 ,150 ,175 ,200 ,225 ,250 ,275 ,300 ,325 ,350 ,375 ,400 ,425 ,450 ,475 ,500 ,525 ,550 ,575 ,600 ,625 ,650 ,675 ,700 ,725 ,750 ,775 ,800 ,825 ,850 ,875 ,900 ,925 ,950 ,975 ,1000]  # 频率
sampling_rate = 200000  # 采样率
duration = 1/100  # 信号时长
rpm = 6000
rps = rpm/60
degps = 360/rps
# %%生成时间序列和角度序列
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
deg = np.linspace(0, 360, int(duration * sampling_rate), endpoint=False)
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
df1 = pd.DataFrame({
    "freq":frequencies,
    "amp":amplitudes,
})
df1.to_csv('Simulation\Data\simSignal1.csv',index=False)
df2 = pd.DataFrame({
    "t":t,
    "deg":deg,
    "mixed_signal":mixed_signal
})
df2.to_csv('Simulation\Data\simSignal2.csv',index=False)
# %%仿真信号波形图像
plt.plot(t, mixed_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Mixed Signal')
plt.show()