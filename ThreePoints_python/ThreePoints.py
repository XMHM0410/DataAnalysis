import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%读取滤波后的数据
df = pd.read_csv('ThreePoints_Python\Filtereddata8.csv')
s1 = df['filtered_s1'].values
s2 = df['filtered_s2'].values
s3 = df['filtered_s3'].values
# %%定义基本参数
rpm = 6000 # 转速
alpha = 90 # 探头12夹角
beta = 90 # 探头34夹角
c1 = 1
c2 = -np.sin(np.deg2rad(alpha+beta))/np.sin(np.deg2rad(beta))
c3 = np.sin(np.deg2rad(alpha))/np.sin(np.deg2rad(beta))
x = s1*c1+s2*c2+s3*c3 # 三点法表达式
N = len(x) # 采样总点数
fs = 2000.0 # 采样频率 Hz
t_total = N/fs
t = np.arange(0,t_total,t_total/N)
# %%绘制原始信号波形图
plt.figure(1)
plt.plot(t[0:1000],x[0:1000]) # 前1000条
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Waveform')
# %%计算频谱
fft_x = np.fft.fft(x)
freq = np.fft.fftfreq(N, d=1/fs)
amp = np.abs(fft_x)
# %%选择感兴趣的频率范围
interest_freq_range = (0.01, 1000.0)  # 单位为Hz
interest_freq_mask = (freq >= interest_freq_range[0]) & (freq <= interest_freq_range[1])
freq_i = freq[interest_freq_mask]
amp_i = amp[interest_freq_mask]
# %%找到最大幅值及对应的频率
max_amp = np.max(amp_i)
max_amp_freq = freq_i[np.argmax(amp_i)]
# %%绘制原始信号频谱幅值谱
plt.figure(2)
# plt.stem(freq[:len(freq)//2], amp[:len(amp)//2])
plt.stem(freq_i, amp_i)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum')
# plt.xlim(interest_freq_range)
#'freq_i':freq_i,'amp_i':amp_i,
# %%按转速基频倍频分离同步误差和异步误差
bf = rpm/60 # 基频 Hz
# bf_index = int(bf*(N/fs)) #转换成索引
bf_index = np.where(freq == np.float64(bf))[0][0] #转换成索引
Sync = fft_x.copy()
Async = fft_x.copy()
Sync[0] = 0 # 把频率为0的第一项去掉，相当于滤掉部分随机误差
Async[0] = 0
for i in range(len(freq_i)):
    if (i) % bf_index == 0:
        Async[i] = 0
    else:
        Sync[i] = 0
# %%同步误差进一步分离圆度误差和偏心误差
Rod = Sync.copy()
Pos = Sync.copy()
Rod[bf_index] = 0 # 圆度误差，去掉第一项
Pos[:bf_index-1] = 0
Pos[bf_index+1:] = 0 # 偏心误差，保留第一项
# %%分离完的信号IFFT
Sync_sig = np.fft.ifft(Sync)
Async_sig = np.fft.ifft(Async)
Rod_sig = np.fft.ifft(Rod)
Pos_sig = np.fft.ifft(Pos)
# %%分离完信号的频谱
Sync_amp = np.abs(Sync)
Async_amp = np.abs(Async)
Rod_amp = np.abs(Rod)
Pos_amp = np.abs(Pos)
# %%误差绘图
"""
# 同步误差
plt.figure(3)
plt.plot(t[0:100],Sync_sig[0:100])
plt.title('Sync')
# 同步误差频谱
plt.figure(4)
plt.stem(freq_i, Sync_amp[interest_freq_mask])
plt.title('Sync_freq')

# 异步误差
plt.figure(5)
plt.plot(t[0:100],Async_sig[0:100])
plt.title('Async')

# 异步误差频谱
plt.figure(6)
plt.stem(freq_i, Async_amp[interest_freq_mask])
plt.title('Async_freq')

# 圆度误差
plt.figure(7)
plt.plot(t[0:100],Rod_sig[0:100])
plt.title('Rod')

# 圆度误差频谱
plt.figure(8)
plt.stem(freq_i, Rod_amp[interest_freq_mask])
plt.title('Rod_freq')

# 偏心误差
plt.figure(9)
plt.plot(t[0:100],Pos_sig[0:100])
plt.title('Pos')

# 偏心误差频谱
plt.figure(10)
plt.stem(freq_i, Pos_amp[interest_freq_mask])
plt.title('Pos_freq')

# 打印所有感兴趣的频率
print('Max AMP :',max_amp,'μm')
print('Max AMP Frequency:',max_amp_freq,'Hz')
print('Sync',Sync_sig)
print('Async',Async_sig)
print('Roundness',Rod_sig)
print('Position',Pos_sig)

# 补：极坐标绘图
theta = np.linspace(0,40*np.pi,1000)# plot1000个点
fig = plt.figure(11)
ax1 = fig.add_subplot(221,projection='polar')
ax1.plot(theta,Sync_sig[10:1010])
ax1.set_title('Sync')
ax2 = fig.add_subplot(222,projection='polar')
ax2.plot(theta,Async_sig[10:1010])
ax2.set_title('Async')
ax3 = fig.add_subplot(223,projection='polar')
ax3.plot(theta,Rod_sig[10:1010])
ax3.set_title('Rod')
ax4 = fig.add_subplot(224,projection='polar')
ax4.plot(theta,Pos_sig[10:1010])
ax4.set_title('Pos')
"""
plt.show()

# %%补：输出到文件
df = pd.DataFrame({'x': x,'Sync': Sync_sig, 'Async': Async_sig, 'Rod': Rod_sig, 'Pos':Pos_sig})
df.to_csv('ThreePoints_Python\Resultdata.csv', index=False)