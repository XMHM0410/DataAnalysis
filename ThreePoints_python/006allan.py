import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Allan方差计算
def allan_variance(data, tau = 10):
    """
    计算Allan方差
    :param data: 数据数组
    :param tau: 时间间隔 默认取10
    :return: Allan方差数组
    """
    n = len(data)
    max_m = int(np.floor(n / (2 * tau)))
    if max_m == 0:
        raise ValueError("tau is too large for the given data array")
    allan_var = np.zeros(max_m)
    for m in range(1, max_m + 1):
        sum_var = 0
        for i in range(1, n - 2 * m * tau + 1):
            denominator = 2 * (n - 2 * m * tau) * tau ** 2
            if denominator == 0:
                raise ValueError("denominator is zero")
            sum_var += (data[i + 2 * m * tau - 1] - 2 * data[i + m * tau - 1] + data[i - 1]) ** 2 / denominator
        allan_var[m - 1] = sum_var
    return allan_var

# %%读文件
df1 = pd.read_csv('ThreePoints_Python\Data\SyncAndAsyncData.csv')
sync = df1['Sync'].values
Async = df1['Async'].values
df2 = pd.read_csv('ThreePoints_Python\Data\denoiceResultData.csv')
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
# %%确定参数
tau = 10
# %%计算分离结果同步异步误差allan方差
sync_allan_var = allan_variance(sync, tau)
sync_x= np.arange(1, len(sync_allan_var) + 1) * tau
async_allan_var = allan_variance(Async, tau)
async_x = np.arange(1, len(async_allan_var) + 1) * tau
# %%文件输出
df_out = pd.DataFrame({
                   'sync_x': sync_x,
                   'sync_allan_var': sync_allan_var,
                   'async_x': async_x,
                   'async_allan_var': async_allan_var,
                   })
df_out.to_csv('ThreePoints_Python\Data\AllanResultData.csv',index=False)