import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import plotThree,plotOne
# %% 测量原始数据
df = pd.read_csv('ThreePoints_CEEMDAN_one\Data\downSample\DownSapleSignal7.csv')
s1 = df["s1"].values
s2 = df["s2"].values
s3 = df["s3"].values
s1 = s1[:6800]# 取前6800个
s2 = s2[:6800]
s3 = s3[:6800]
# %% 定义基本参数
rpm = 12000 # 转速
alpha = 84 # 探头12夹角 84
beta = 175-84 # 探头34夹角 91
fs = 1360000.0 # 采样频率 Hz
# 权系数c1 c2 c3
c1 = 1
c2 = -np.sin(np.deg2rad(alpha+beta))/np.sin(np.deg2rad(beta))
c3 = np.sin(np.deg2rad(alpha))/np.sin(np.deg2rad(beta))
print("c1:",c1,"c2:",c2,"c3:",c3)
# 生成时间和角度轴信息
N = len(s1) # 采样总点数
t_total = N/fs
t = np.arange(0,t_total,t_total/N) #生成时间轴
theta = np.arange(0,360,360/N) #生成角度轴
# %%分别乘系数变成合理的数值并添加噪声
plotThree.plotThree(theta,s1,s2,s3)
# np.random.seed(123)  种子会使每次运行生成的随机数相同
noise1 = np.random.normal(scale=0.05,size=N) # 生成高斯白噪声
s1 = s1*20 + noise1
noise2 = np.random.normal(scale=0.05,size=N)
s2 = s2*20 + noise2
noise3 = np.random.normal(scale=0.05,size=N)
s3 = s3*20 + noise3
plotThree.plotThree(theta,s1,s2,s3)
# %%三点法计算
S = s1*c1+s2*c2+s3*c3 # 三点法表达式
plotOne.plotOne(theta,S)
plt.show()
# %%文件保存
df1 = pd.DataFrame({"t":t,"theta":theta,"S":S,'s1':s1,'s2':s2,'s3':s3})
df1.to_csv('ThreePoints_CEEMDAN_one\Data\MixedSignal.csv', index=False)
# 用7
df2 = pd.DataFrame({"rpm":[rpm],"alpha":[alpha],"beta":[beta],"fs":[fs],"c1":[c1],"c2":[c2],"c3":[c3]})
df2.to_csv('ThreePoints_CEEMDAN_one\Data\config.csv')