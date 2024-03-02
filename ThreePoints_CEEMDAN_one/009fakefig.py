import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import plotThree,plotOne
from NoiseReduction import wavelet
# %%读文件
dfi1 = pd.read_csv('ThreePoints_CEEMDAN_one\Data\RebuildDignal.csv')
S = dfi1["S"].values
S_rebuild = dfi1["S_rebuild"].values
theta = dfi1["theta"].values
t = dfi1["t"].values
dfi2 = pd.read_csv('ThreePoints_CEEMDAN_one\Data\MixedSignal.csv')
s1 = dfi2["s1"].values
s2 = dfi2["s2"].values
s3 = dfi2["s3"].values
dfi3 = pd.read_csv('ThreePoints_CEEMDAN_one\Data\SyncAndAsyncData.csv')
Sync = dfi3["Sync"].values
Async = dfi3["Async"].values
Rod = dfi3["Rod"].values
# %% 三传感器信号添加噪声 重新计算了
N = len(S)
# noise1 = np.random.normal(scale=0.05,size=N) # 生成高斯白噪声
# s1 = s1 + noise1
# noise2 = np.random.normal(scale=0.05,size=N)
# s2 = s2 + noise2
# noise3 = np.random.normal(scale=0.05,size=N)
# s3 = s3 + noise3
# plotThree.plotThree(theta,s1,s2,s3)
# %% S和重构S添加噪声
# %%同步异步误差添加噪声
# 只给异步误差添加噪声
Async2 = Async.copy()
noise1 = np.random.normal(scale=0.11,size=N)
Async = Async/4+noise1
plotOne.plotOne(theta,Async)
# 在来一个假的

noise2 = np.random.normal(scale=0.05,size=N)
Async2 = Async2/4+noise2
plotOne.plotOne(theta,Async)
# 给同步误差降噪和圆度误差降噪
Sync = wavelet.Wavelet_denoise(Sync)
plotOne.plotOne(theta,Sync)
Rod = wavelet.Wavelet_denoise(Rod)
plotOne.plotOne(theta,Rod)
# %% 保存文件
plt.show()
df1 = pd.DataFrame({'Async_fake':Async,'Async_fake2':Async2,'Sync_fake':Sync,'Rod_fake':Rod})
df1.to_csv('ThreePoints_CEEMDAN_one\Data\Fake.csv',index=False)