import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %% 测量原始数据
df = pd.read_csv('ThreePoints_CEEMDAN_one\Data\downSample\DownSapleSignal3.csv')
s1 = df["s1"].values
s2 = df["s2"].values
s3 = df["s3"].values
# %% 定义基本参数
rpm = 12000 # 转速
alpha = 84 # 探头12夹角
beta = 175-84 # 探头34夹角
# 权系数
c1 = 1
c2 = -np.sin(np.deg2rad(alpha+beta))/np.sin(np.deg2rad(beta))
c3 = np.sin(np.deg2rad(alpha))/np.sin(np.deg2rad(beta))
S = s1*c1+s2*c2+s3*c3 # 三点法表达式
N = len(S) # 采样总点数
fs = 20000.0 # 采样频率 Hz
t_total = N/fs
t = np.arange(0,t_total,t_total/N)
theta = np.arange(0,360,360/N)
# %%plot S
plt.figure()
plt.plot(t,S)
plt.xlabel('Time (s)')
plt.ylabel('S (mm)')
plt.title('Mixed Signal')
plt.show()
# %%文件保存
df1 = pd.DataFrame({"t":t,"theta":theta,"S":S})
df1.to_csv('ThreePoints_CEEMDAN_one\Data\downSample\MixedSignal3.csv', index=False)
# 用6