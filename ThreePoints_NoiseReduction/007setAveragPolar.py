import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%读文件
df1 = pd.read_csv('ThreePoints_NoiseReduction\Data\SyncAndAsyncData.csv')
sync = df1['Sync'].values
Async = df1['Async'].values
df2 = pd.read_csv('ThreePoints_NoiseReduction\Data\denoiceResultData.csv')
x = df2['x'].values
t = df2['t'].values
sync_kalman = df2['sync_kalman'].values
async_kalman = df2['async_kalman'].values
sync_wavelet = df2['sync_wavelet'].values
async_wavelet = df2['async_wavelet'].values
# sync_LSTMGRU = df2['sync_LSTMGRU'].values
# async_LSTMGRU = df2['async_LSTMGRU'].values
sync_average = df2['sync_average'].values
async_average = df2['async_average'].values
df22 = pd.read_csv('ThreePoints_Python\Data\denoiseExtra.csv')
sync_wavelet_extra = df22['sync_wavelet_extra'].values
async_wavelet_extra = df22['async_wavelet_extra'].values
# %%采样信号基础信息
rpm = 6000 # 转速
N = len(x) # 采样总点数
fs = 2000.0 # 采样频率 Hz
t_total = N/fs
rps = rpm/60
deg = np.arange(0,360,0.18)
# %%集合平均求误差圆

# %%求误差上下限圆
sync_max = np.max(sync)
sync_wavelet_max = np.max(sync_wavelet_extra)
async_max = np.max(Async)
async_wavelet_max = np.max(async_wavelet_extra)
sync_min = np.min(sync)
sync_wavelet_min = np.min(sync_wavelet_extra)
async_min = np.min(Async)
async_wavelet_min = np.min(async_wavelet_extra)
print(sync_max,sync_min)
print(sync_wavelet_max,sync_wavelet_min)
print(async_max,async_min)
print(async_wavelet_max,async_wavelet_min)
# %%求误差最小二乘圆

# %%保存文件
df3 = pd.DataFrame({
    "deg":deg,
})
df3.to_csv('ThreePoints_NoiseReduction\Data\ErrorCircle.csv',index=False)