import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def comb_filter(input_signal, sampling_rate, frequency, q_factor):
    # 计算参数
    delay = int(sampling_rate / frequency)  # 延迟长度
    damping = 1 / q_factor  # 衰减因子
    # 创建延迟线
    delay_line = np.zeros(delay)
    output_signal = np.zeros(len(input_signal))
    # 滤波处理
    for i, x in enumerate(input_signal):
        output_signal[i] = x + damping * delay_line[0]  # 输出信号为输入信号加上延迟线上的信号
        # 更新延迟线
        delay_line = np.roll(delay_line, 1)  # 延迟线左移一位
        delay_line[0] = output_signal[i]  # 更新延迟线的第一个元素
    return output_signal

# 示例使用
sampling_rate = 2000  # 采样率
frequency = 25  # 梳状滤波器的频率
q_factor = 30  # 衰减因子
duration = 5  # 输入信号的时长

# 生成输入信号（示例：440Hz正弦波）
time = np.arange(0, duration, 1/sampling_rate)
input_signal = np.sin(2 * np.pi * 50 * time)+np.sin(2 * np.pi * 25 * time)

# 应用梳状滤波器
output_signal = comb_filter(input_signal, sampling_rate, frequency, q_factor)
# 绘制输入输出信号
plt.plot(time[0:1000], input_signal[0:1000], label='input_signal')
plt.plot(time[0:1000], output_signal[0:1000], label='output_signal')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.legend()
plt.show()