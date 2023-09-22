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
# %%输出到文件
df = pd.DataFrame({'x': x,'Sync': abs(Sync_sig), 'Async': abs(Async_sig), 'Rod': abs(Rod_sig), 'Pos': abs(Pos_sig)})
df.to_csv('ThreePoints_Python\Data\Resultdata.csv', index=False)
# %%误差绘图