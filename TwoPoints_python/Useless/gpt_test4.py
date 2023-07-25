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

# 设置感兴趣的频率范围
interest_freq_range = (1, 20)

# 找到感兴趣的频率范围内的峰值
interest_freq_mask = np.logical_and(freq >= interest_freq_range[0], freq <= interest_freq_range[1])
max_freq_idx = np.argmax(psd_signal1[interest_freq_mask])

# 计算相位差
phase_diff = np.angle(fft_signal1) - np.angle(fft_signal2)

# # 计算同步误差
sync_phase_mask = np.logical_or(phase_diff == 0, phase_diff == np.pi)
# sync_amp_ratio = np.abs(fft_signal1[interest_freq_mask][sync_phase_mask]) / np.abs(fft_signal2[interest_freq_mask][sync_phase_mask])
# sync_contour = np.real(np.fft.ifft(fft_signal2 * sync_amp_ratio))

###############################################################################
# 计算感兴趣的频率掩码和同步相位掩码的索引
interest_freq_idx = np.where(interest_freq_mask)[0]
sync_phase_idx = np.where(np.logical_or(phase_diff == 0, phase_diff == np.pi))[0]

# 切片 fft_signal1 和 fft_signal2
fft_signal1_slice = fft_signal1[interest_freq_mask][sync_phase_mask]
fft_signal2_slice = fft_signal2[interest_freq_mask][sync_phase_mask]

# 计算同步误差
sync_amp_ratio = np.abs(fft_signal1_slice) / np.abs(fft_signal2_slice)
sync_contour = np.real(np.fft.ifft(fft_signal2 * sync_amp_ratio))
################################################################################

# 计算同步误差的幅值和相位差
sync_error = contour - sync_contour
sync_amp = np.max(sync_error) - np.min(sync_error)
sync_phase = -np.angle(np.fft.fft(sync_error)[max_freq_idx])

# 计算异步误差
async_phase_mask = np.logical_and(phase_diff != 0, phase_diff != np.pi)
async_amp_ratio = np.abs(fft_signal1[interest_freq_mask][async_phase_mask]) / np.abs(fft_signal2[interest_freq_mask][async_phase_mask])
async_contour = np.real(np.fft.ifft(fft_signal2 * async_amp_ratio))

# 计算异步误差的幅值和相位差
async_error = contour - async_contour
async_amp = np.max(async_error) - np.min(async_error)
async_phase = -np.angle(np.fft.fft(async_error)[max_freq_idx])

# 计算工件圆度误差
roundness_error = sync_contour - async_contour
roundness_amp = np.max(roundness_error) - np.min(roundness_error)

# 绘制结果图
fig, ax = plt.subplots(4, 1, figsize=(8, 10))
ax[0].plot(theta, contour)
ax[0].set_title('Contour')
ax[1].plot(theta, signal1, label='Signal 1')
ax[1].plot(theta, signal2, label='Signal 2')
ax[1].set_title('Signals')
ax[1].legend()
ax[2].plot(freq, psd_signal1, label='Signal 1')
ax[2].plot(freq, psd_signal2, label='Signal 2')
ax[2].set_xlim(interest_freq_range)
ax[2].set_title('Power Spectral Density')
ax[2].legend()
ax[3].plot(theta, sync_contour, label='Sync Error')
ax[3].plot(theta, async_contour, label='Async Error')
ax[3].plot(theta, contour, label='Roundness Error')
ax[3].set_title('Errors')
ax[3].legend()
plt.show()
