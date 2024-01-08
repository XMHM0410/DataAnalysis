import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from noisefiltering import wavelet,kalman,average
# %%读文件
df1 = pd.read_csv('ThreePoints_NoiseReduction\Data\SyncAndAsyncData.csv')
sync = df1['Sync'].values
Async = df1['Async'].values
df2 = pd.read_csv('ThreePoints_NoiseReduction\Data\denoiceResultData.csv')
x = df2['x'].values
t = df2['t'].values
sync_wavelet = df2['sync_wavelet'].values
async_wavelet = df2['async_wavelet'].values
# %%小波降噪*2
sync_wavelet_extra = wavelet.Wavelet_denoise(sync_wavelet)
async_wavelet_extra = wavelet.Wavelet_denoise(async_wavelet)
# %%再集合平均20次
i = 1
for i in range(0,20):
    sync_wavelet_extra = average.average_denoise(sync_wavelet_extra)
    async_wavelet_extra = average.average_denoise(async_wavelet_extra)
    i += 1
# %%保存文件
df3 = pd.DataFrame({
    "t":t,
    "sync_wavelet_extra":sync_wavelet_extra,
    "async_wavelet_extra":async_wavelet_extra,
})
df3.to_csv('ThreePoints_NoiseReduction\Data\denoiseExtra.csv',index=False)