import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from noisefiltering import wavelet,kalman,average
from allan import allan
# %%读文件
# 同步异步误差分离数据
df1 = pd.read_csv('ThreePoints_NoiseReduction\Data\SyncAndAsyncData.csv')
sync = df1['Sync'].values
Async = df1['Async'].values
# 读取小波阈值降噪结果
df2 = pd.read_csv('ThreePoints_NoiseReduction\Data\denoiceResultData.csv')
x = df2['x'].values
t = df2['t'].values
sync_wavelet = df2['sync_wavelet'].values
async_wavelet = df2['async_wavelet'].values
# %%更改参数的小波阈值降噪
# 同步误差小波
sync_wavelet2 = wavelet.Wavelet_denoise(sync,wavelet = 'db3', level = 2)
# 异步误差小波
async_wavelet2 = wavelet.Wavelet_denoise(Async,wavelet = 'db3', level = 2)
# %%为原小波阈值结果添加白噪声
# 为sync_wavelet2添加高斯白噪声
noise = np.random.normal(scale=0.0003,size=sync_wavelet2.shape) # 生成高斯白噪声
sync_wavelet2 = sync_wavelet2 + noise # 将高斯白噪声乘以信号的平方根
# 为async_wavelet2添加高斯白噪声
async_wavelet2 = async_wavelet2 + noise # 将高斯白噪声乘以信号的平方根
# %%计算新的Allan方差
tau = 10
sync_wavelet2_allan_var = allan.allan_variance(sync_wavelet2, tau)
sync_wavelet2_x= np.arange(1, len(sync_wavelet2_allan_var) + 1) * tau
async_wavelet2_allan_var = allan.allan_variance(async_wavelet2, tau)
async_wavelet2_x = np.arange(1, len(async_wavelet2_allan_var) + 1) * tau
# %%输出到文件
df_out1 = pd.DataFrame({
                    'x': x,
                    't': t,
                    'sync_wavelet2': sync_wavelet2,
                    'async_wavelet2': async_wavelet2
})
df_out1.to_csv('ThreePoints_NoiseReduction\Data\AddDenoiseData.csv',index=False)
df_out2 = pd.DataFrame({
                   'sync_wavelet2_x': sync_wavelet2_x,
                   'sync_wavelet2_allan_var': sync_wavelet2_allan_var,
                   'async_wavelet2_x': async_wavelet2_x,
                   'async_wavelet2_allan_var': async_wavelet2_allan_var,
                   })
df_out2.to_csv('ThreePoints_NoiseReduction\Data\AddAllanResultData.csv',index=False)