import numpy as np
import matplotlib.pyplot as plt
from pyemd import EMD

# 假设采样频率为f，数组为waveform
f = 1000
t = np.arange(0, 1, 1/f)
waveform = np.sin(2 * np.pi * 10 * t)  # 生成频率为10Hz的正弦波

# 创建EMD对象
emd = EMD()

# 进行经验模态分解
IMFs = emd.emd(waveform)

# 绘制每个IMF的频谱
for i, IMF in enumerate(IMFs):
    # 计算频谱
    spectrum = np.fft.fft(IMF)
    amplitude = np.abs(spectrum)
    
    # 绘制频谱图
    plt.subplot(len(IMFs), 1, i+1)
    plt.plot(amplitude)
    plt.title("IMF {}".format(i+1))

plt.show()

