# 在这个例子中，我们首先使用numpy.loadtxt()函数从文件中读取信号数据，然后使用numpy.fft.fft()函数计算傅里叶变换。我们还使用numpy.fft.fftfreq()函数计算出对应的频率数组。
# 接下来，我们计算信号的功率谱密度（PSD），并根据感兴趣的频率范围选择出相应的PSD值和频率值。然后我们找到最大PSD值对应的频率，并计算出主轴回转误差的幅值比和相位差。最后，我们使用matplotlib.pyplot库绘制信号的频谱图。
# 需要注意的是，这个示例只是一个简单的频域两点法算法示例，实际的算法可能会更加复杂，需要根据具体的需求进行调整和优化。
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

# 读取信号数据
data = pd.read_csv('originaldata.txt', header=None, delimiter='\t')
signal_data = data.iloc[:, 1].values

# 指定采样频率
sampling_rate = 1000.0

# 计算傅里叶变换
fft_data = np.fft.fft(signal_data)
freq = np.fft.fftfreq(signal_data.shape[-1], d=1/sampling_rate)

# 获取信号的功率谱密度
psd = np.abs(fft_data) ** 2 / len(fft_data)

# 选择感兴趣的频率范围
interest_freq_range = (0.5, 200.0)  # 单位为Hz
interest_freq_mask = (freq >= interest_freq_range[0]) & (freq <= interest_freq_range[1])

# 找到最大幅值对应的频率
max_psd_freq = freq[interest_freq_mask][np.argmax(psd[interest_freq_mask])]

# 计算主轴回转误差幅值和相位差
amp_ratio = np.abs(fft_data[interest_freq_mask][np.argmax(psd[interest_freq_mask])]) / np.abs(fft_data[0])
phase_diff = np.angle(fft_data[interest_freq_mask][np.argmax(psd[interest_freq_mask])]) - np.angle(fft_data[0])

# 输出结果
print('Max PSD Frequency:', max_psd_freq, 'Hz')
print('Amplitude Ratio:', amp_ratio)
print('Phase Difference:', phase_diff, 'rad')

# 绘制信号的频谱图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(freq[interest_freq_mask], psd[interest_freq_mask], 'r')
ax.set_xlim(interest_freq_range)
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title('Spindle Runout Error Spectrum')
plt.show()
