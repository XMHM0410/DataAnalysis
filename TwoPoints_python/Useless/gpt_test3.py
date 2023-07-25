# 在这个示例中，我们使用numpy.linspace()函数生成一个轮廓曲线，并通过numpy.random.normal()函数生成一个噪音信号。
# 然后，我们合成两个仿真信号（signal1和signal2），并分别进行傅里叶变换。接下来，我们计算信号的功率谱密度，并选择感兴趣的频率范围。
# 最后，我们找到最大PSD值对应的频率，并计算出主轴回转误差的幅值比和相位差。最终，我们使用matplotlib.pyplot库绘制信号的
import numpy as np
import matplotlib.pyplot as plt

# 生成轮廓曲线
theta = np.linspace(0, 2 * np.pi, 1000)
contour = 10 * np.cos(5 * theta)

# 生成噪音信号
np.random.seed(0)
noise = np.random.normal(0, 0.5, len(theta))

# 合成两个仿真信号
signal1 = contour + noise
signal2 = contour + 0.5 * noise

# 计算采样频率
sampling_rate = 1000.0
dt = 1 / sampling_rate

# 计算傅里叶变换
fft_signal1 = np.fft.fft(signal1)
fft_signal2 = np.fft.fft(signal2)
freq = np.fft.fftfreq(len(signal1), dt)

# 获取信号的功率谱密度
psd_signal1 = np.abs(fft_signal1) ** 2 / len(fft_signal1)
psd_signal2 = np.abs(fft_signal2) ** 2 / len(fft_signal2)

# 选择感兴趣的频率范围
interest_freq_range = (0.5, 100.0)  # 单位为Hz
interest_freq_mask = (freq >= interest_freq_range[0]) & (freq <= interest_freq_range[1])

# 找到最大幅值对应的频率
max_psd_freq1 = freq[interest_freq_mask][np.argmax(psd_signal1[interest_freq_mask])]
max_psd_freq2 = freq[interest_freq_mask][np.argmax(psd_signal2[interest_freq_mask])]

# 计算主轴回转误差幅值和相位差
amp_ratio = np.abs(fft_signal1[interest_freq_mask][np.argmax(psd_signal1[interest_freq_mask])]) / np.abs(fft_signal2[interest_freq_mask][np.argmax(psd_signal2[interest_freq_mask])])
phase_diff = np.angle(fft_signal1[interest_freq_mask][np.argmax(psd_signal1[interest_freq_mask])]) - np.angle(fft_signal2[interest_freq_mask][np.argmax(psd_signal2[interest_freq_mask])])

# 输出结果
print('Max PSD Frequency Signal1:', max_psd_freq1, 'Hz')
print('Max PSD Frequency Signal2:', max_psd_freq2, 'Hz')
print('Amplitude Ratio:', amp_ratio)
print('Phase Difference:', phase_diff, 'rad')

# 绘制信号的频谱图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(freq[interest_freq_mask], psd_signal1[interest_freq_mask], 'r', label='Signal1')
ax.plot(freq[interest_freq_mask], psd_signal2[interest_freq_mask], 'b', label='Signal2')
ax.set_xlim(interest_freq_range)
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title('Spindle Runout Error Spectrum')
ax.legend()
plt.show()
