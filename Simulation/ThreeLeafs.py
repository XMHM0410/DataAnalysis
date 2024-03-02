import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 思路：一圈360°，作为一个周期，3456叶图周期为x分之一
# %%定义正弦波的参数
# amplitudes =  [5.0,0.8,0.6,0.4,0.2,0.1,0.1,0.1,0.1,0.08,0.06,0.04 ,0.02,0.01]  # 振幅
# frequencies = [1 ,10,12,20,25,30,36,40,50,60 ,70 ,80 ,90 ,100]  # 频率
amplitudes =  [4.0,0.2,0.1,0.1]  # 振幅
frequencies = [10,80,100,200]  # 频率
sampling_rate = 2000  # 采样率
duration = 1  # 信号时长
rpm = 6000
# %%生成时间序列和角度序列
rps = rpm/60
degps = 360/rps
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
deg = np.linspace(0, 360, int(duration * sampling_rate), endpoint=False)
# %%生成正弦波信号
signals = []
for amplitude, frequency in zip(amplitudes, frequencies):
    signal = amplitude * np.cos(2 * np.pi * frequency * t)
    signals.append(signal)
# %%合并信号
mixed_signal = np.sum(signals, axis=0)
# %%添加高斯白噪音
noise = np.random.normal(0, 0.01, len(t))
mixed_signal = mixed_signal + noise
# %%创建DataFrame并保存为.csv文件
# %%保存文件
df3 = pd.DataFrame({
    "t":t,
    "deg":deg,
    "mixed_signal":mixed_signal
})
df3.to_csv('Simulation\Data\simSignal.csv',index=False)
# %%仿真信号波形图像
plt.plot(deg, mixed_signal)
plt.xlabel('deg')
plt.ylabel('amp')
plt.show()