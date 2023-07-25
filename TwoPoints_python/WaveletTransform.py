import pywt
import numpy as np
import matplotlib.pyplot as plt

# 生成包含噪声信号的数据
t = np.linspace(0, 1, 200, endpoint=False)
signal = np.sin(4 * np.pi * t) + np.random.randn(200) * 0.6

# 小波变换
wavelet = 'db4'  # 选择小波基函数
level = 2        # 小波分解的级别
coefficients = pywt.wavedec(signal, wavelet, level=level)

# 阈值处理
sigma = np.median(np.abs(coefficients[-level])) / 0.6745   # 计算噪声标准差
threshold = sigma * np.sqrt(2 * np.log(len(signal)))       # 计算阈值
coefficients[1:] = (pywt.threshold(i, value=threshold, mode="soft") for i in coefficients[1:])

# 反小波变换
reconstructed_signal = pywt.waverec(coefficients, wavelet)

# 绘制结果
plt.plot(t, signal, label='Original signal')
plt.plot(t, reconstructed_signal, label='Denoised signal')
plt.legend()
plt.show()
