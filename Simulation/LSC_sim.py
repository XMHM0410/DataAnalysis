import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %% 最小二乘圆法仿真
# 思路：生成一组数（-8~8）,计算平均值，计算最大值，计算最小值，输出
sampling_rate = 180  # 采样率
duration = 1  # 信号时长
rpm = 6000
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
deg = np.linspace(0, 360, int(duration * sampling_rate), endpoint=False)
np.random.seed(6)
noise = np.random.normal(0, 1, len(t))
signal = noise+2
# 计算平均值
avg = np.mean(signal)
# 将noise从小到大排序
sorted_signal = np.sort(signal)
# 取其中最小的18个数
noise_min = sorted_signal[:18]
# 取其中最大的18个数
noise_max = sorted_signal[-18:]
# 计算最小的18个数的平均值
min_val = np.mean(noise_min)
# 计算最大的18个数的平均值
max_val = np.mean(noise_max)
print("平均值：", avg)
print("区间最大值：", max_val)
print("区间最小值：", min_val)
# %%导出文件
df3 = pd.DataFrame({
    "t":t,
    "deg":deg,
    "mixed_signal":signal
})
df3.to_csv('Simulation\Data\LSC_Signal.csv',index=False)