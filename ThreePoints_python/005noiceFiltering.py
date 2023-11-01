import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from noisefiltering import lstmgru,wavelet,kalman,average

# %%读取同步异步误差
df = pd.read_csv('ThreePoints_Python\Data\SyncAndAsyncData.csv')
sync = df['Sync'].values
Async = df['Async'].values
t = df['t'].values
x = df['x'].values
# %%同步误差卡尔曼滤波
sync_kalman = kalman.Kalman_denoise(sync)
# %%异步误差卡尔曼滤波
async_kalman = kalman.Kalman_denoise(Async)
# %%同步误差小波包
sync_wavelet = wavelet.Wavelet_denoise(sync)
# %%异步误差小波包
async_wavelet = wavelet.Wavelet_denoise(Async)
# %%同步误差LSTM-GRU
sync_LSTMGRU = []
# %%异步误差LSTM-GRU
async_LSTMGRU = []
# %%同步误差集合平均
sync_average = average.average_denoise(sync)
# %%异步误差集合平均
async_average = average.average_denoise(Async)
# %%降噪结果输出
df_out = pd.DataFrame({'x': x,
                   't': t,
                   'sync_kalman': sync_kalman,
                   'async_kalman': async_kalman,
                   'sync_wavelet': sync_wavelet,
                   'async_wavelet': async_wavelet,
                #    'sync_LSTMGRU': sync_LSTMGRU,
                #    'async_LSTMGRU': async_LSTMGRU,
                   'sync_average': sync_average,
                   'async_average': async_average
                   })
df_out.to_csv('ThreePoints_Python\Data\denoiceResultData.csv',index=False)