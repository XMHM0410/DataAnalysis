import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from EMD import ceemdan
# %%读文件三点法混合信号
df = pd.read_csv('ThreePoints_EMD_Improve\Data\MixedSignal.csv')
S = df["S"].values
t = df["t"].values
# %%CEEMDAN分解
# elec_all_day_test = x[0:2000]
elec_all_day_test = S.copy()
IImfs,res=ceemdan.ceemdan_decompose(np.array(elec_all_day_test).ravel())
# # IMF为IMF分量矩阵，res为分解残余量，也叫重建误差
output = {"t":t,"S":S,"res":res}
for i in range(IImfs.shape[0]):
    output[f"IMF{i+1}"] = IImfs[i]
# %%文件保存
df1 = pd.DataFrame(output)
df1.to_csv('ThreePoints_EMD_Improve\Data\IMFs.csv', index=False)