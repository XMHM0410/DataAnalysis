from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fs = 2000.0  # 采样频率
f0 = 100.0 # 要去除的频率
Q = 30.0 #品质因数
# %%陷波梳状滤波器
b,a = signal.iircomb(f0,Q,ftype='notch',fs=fs)
# %%频响
# Frequency response
freq, h = signal.freqz(b, a, fs=fs)
response = abs(h)
# To avoid divide by zero when graphing
response[response == 0] = 1e-20
# Plot
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(freq, 20*np.log10(abs(response)), color='blue')
ax[0].set_title("Frequency Response")
ax[0].set_ylabel("Amplitude (dB)", color='blue')
ax[0].set_xlim([0, 1000])
ax[0].set_ylim([-30, 10])
ax[0].grid()
ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
ax[1].set_ylabel("Angle (degrees)", color='green')
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_xlim([0, 1000])
ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[1].set_ylim([-90, 90])
ax[1].grid()
plt.show()