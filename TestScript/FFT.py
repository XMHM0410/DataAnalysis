import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('originaldata6-5.txt', header=None, delimiter='\t')
x = data.iloc[:, 0].values
# t = data.iloc[:, 0].values
t = np.arange(0,10000,1)

# 指定采样频率
sampling_rate = 100.0

# 计算频谱
fft_x = np.fft.fft(x)
freq = np.fft.fftfreq(len(x), d=1/sampling_rate)
amp = np.abs(fft_x)

# 计算功率谱
power = np.abs(fft_x)**2 / len(fft_x)

# 计算功率谱密度
psd = power / (freq[1] - freq[0])

# 选择感兴趣的频率范围
interest_freq_range = (0.01, 50.0)  # 单位为Hz
interest_freq_mask = (freq >= interest_freq_range[0]) & (freq <= interest_freq_range[1])

# 找到最大幅值及对应的频率
max_amp = np.max(amp[interest_freq_mask])
max_amp_freq = freq[interest_freq_mask][np.argmax(amp[interest_freq_mask])]

# 找出其他较大幅值及对应的频率
threshold = 5# 幅值阈值
largeAmplitudes = []
for i in range(len(freq[interest_freq_mask])):
    if amp[interest_freq_mask][i] > threshold:
        frequency = freq[interest_freq_mask][i]
        largeAmplitudes.append((frequency, amp[interest_freq_mask][i]))
def list_to_matrix(lst):# 
    matrix = []
    for tup in lst:
        matrix.append([tup[0], tup[1]])
    return matrix
largeAmplitudesMatrix = list_to_matrix(largeAmplitudes)
# 将感兴趣的频率及其对应幅值写入文件
df = pd.DataFrame(largeAmplitudesMatrix, columns=['Frequency', 'Amplitude'])
df.to_csv('freq.csv', index=False)

# 绘制原始信号波形图
plt.figure(1)
plt.plot(t[0:1000],x[0:1000])
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Waveform')
# plt.xlim((0,1000))
# 绘制频谱幅值谱
plt.figure(2)
# plt.stem(freq[:len(freq)//2], amp[:len(amp)//2])
plt.stem(freq[interest_freq_mask], amp[interest_freq_mask])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum')
# plt.xlim(interest_freq_range)

# 绘制功率谱
plt.figure(3)
# plt.plot(freq[:len(freq)//2], power[:len(power)//2])
plt.plot(freq[interest_freq_mask], power[interest_freq_mask])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum')
# plt.xlim(interest_freq_range)

# 绘制功率谱密度
plt.figure(4)
# plt.plot(freq[:len(freq)//2], psd[:len(power)//2])
plt.plot(freq[interest_freq_mask], psd[interest_freq_mask])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density')
# plt.xlim(interest_freq_range)

# 打印所有感兴趣的频率
print('All AMP Freqency',largeAmplitudesMatrix)
print('Max AMP :',max_amp,'μm')
print('Max AMP Frequency:',max_amp_freq,'Hz')



plt.show()

