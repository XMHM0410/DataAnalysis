import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 思路：一圈360°，作为一个周期，3456叶图周期为x分之一
# %%定义正弦波的参数
# amplitudes =  [5.0,0.8,0.6,0.4,0.2,0.1,0.1,0.1,0.1,0.08,0.06,0.04 ,0.02,0.01]  # 振幅
# frequencies = [1 ,10,12,20,25,30,36,40,50,60 ,70 ,80 ,90 ,100]  # 频率
amplitudes =  [4.0,0.2,0.1,0.1]  # 振幅
frequencies = [100,200,300,400]  # 频率
sampling_rate = 2000  # 采样率
duration = 1  # 信号时长
rpm = 6000
# %%生成时间序列和角度序列
rps = rpm/60
degps = 360/rps
# 生成时间和角度轴信息
N = sampling_rate*duration # 采样总点数
t_total = N/sampling_rate # 采样总时长
deg = 360*((rpm/60)*t_total)# 该段时间内转过的角度
t = np.arange(0,t_total,t_total/N) #生成时间轴
theta = np.arange(0,deg,deg/N) #生成角度轴 取10圈的数据
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
    "theta":theta,
    "mixed_signal":mixed_signal
})
df3.to_csv('Simulation\Data\HarSimSignal.csv',index=False)
# %%仿真信号波形图像
plt.plot(theta, mixed_signal)
plt.xlabel('theta')
plt.ylabel('amp')
plt.show()