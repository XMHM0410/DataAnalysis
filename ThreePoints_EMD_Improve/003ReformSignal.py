import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %% 读将采样率后数据
df = pd.read_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal12.csv')
s1 = df["s1_ds12"].values
s2 = df["s2_ds12"].values
s3 = df["s3_ds12"].values
s1 = s1[0:200]
s2 = s2[0:200]
s3 = s3[0:200]
# %%为了可以让数据收尾相接，需要对端点做点处理
s3[197] = s3[197]*10
s3[198] = s3[198]*100 
s3[199] = -s3[199]*10
# %% 将采样率数据前100条重复叠加500次
s1_reform = np.concatenate([s1]*500,axis=0)
s2_reform = np.concatenate([s2]*500,axis=0)
s3_reform = np.concatenate([s3]*500,axis=0)
print(type(s1_reform),s1_reform.shape)
# %%添加高斯白噪声作为误差信号
noise = np.random.normal(scale=0.0005,size=s1_reform.shape) # 生成高斯白噪声
s1_reform = s1_reform + noise
s2_reform = s2_reform + noise
s3_reform = s3_reform + noise
# %%plot
# %%plot
def plotds(s1_ds,s2_ds,s3_ds):
    plt.figure(figsize=(12,6))
    plt.subplot(311)
    plt.plot(s1_ds)
    plt.subplot(312)
    plt.plot(s2_ds)
    plt.subplot(313)
    plt.plot(s3_ds)
    plt.tight_layout()
plotds(s1_reform,s2_reform,s3_reform)
plt.show()
# %%文件保存
df1 = pd.DataFrame({"s1_reform":s1_reform,"s2_reform":s2_reform,"s3_reform":s3_reform})
df1.to_csv('ThreePoints_EMD_Improve\Data\ReformSignal.csv', index=False)