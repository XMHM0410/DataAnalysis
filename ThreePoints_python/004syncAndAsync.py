import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%读取滤波后的数据
df = pd.read_csv('ThreePoints_Python\Data\ThreePointsResultData.csv')
x = df['x'].values
# %%定义基本参数
rpm = 6000 # 转速
N = len(x) # 采样总点数
fs = 2000.0 # 采样频率 Hz
t_total = N/fs
t = np.arange(0,t_total,t_total/N)
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
# %%按转速基频倍频分离同步误差和异步误
bf = rpm/60 # 基频 Hz
bf_index = np.where(freq == np.float64(bf))[0][0] #转换成索引
# bf_index_range = 5 #去基频索引+-5个
# bf_index_range_list = np.arange(bf_index - bf_index_range, bf_index + bf_index_range)
Sync = fft_x.copy()
Async = fft_x.copy()
Sync[0] = 0 # 把频率为0的第一项去掉，相当于滤掉部分随机误差
Async[0] = 0
"""
for i in range(len(freq_i)):
    if (i) % bf_index == 0:
        Async[i] = 0
    else:
        Sync[i] = 0
"""
for i in range(1,len(freq_i)):
    if i % bf_index == 0: #倍频区间
        #取sync[i]前后50个的最大值
        # print(np.arange(i-50,i+50))
        Sync[i] = max(Sync[i-50:i+50])
        # Sync[i] = 0 
        #Async[i]前后50个均置为0
        Async[i-50:i+50] = 0
        # Async[i] = 0      
    # else: #非倍频区间
for j in range(1,len(freq_i)):
    if j % bf_index == 0: #倍频区间
        pass
    else:
        Sync[j] = 0   
for k in range(1,len(freq_i)):
    if Async[k] <= 0.2:
        Async[k] = 0
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
# %%输出到文件
df = pd.DataFrame({'x': x,'Sync': abs(Sync_sig), 'Async': abs(Async_sig), 'Rod': abs(Rod_sig), 'Pos': abs(Pos_sig)})
df.to_csv('ThreePoints_Python\Data\Resultdata.csv', index=False)
# %%误差绘图
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
plt.show()