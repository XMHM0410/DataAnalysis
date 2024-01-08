import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from EMD import ceemdan
# %%读文件三点法混合信号
df = pd.read_csv('ThreePoints_EMD_Improve\Data\ThreePointsResultData.csv')
x = df["x"].values
# %%CEEMDAN分解
elec_all_day_test = x[0:2000]
IImfs,res=ceemdan.ceemdan_decompose(np.array(elec_all_day_test).ravel())
print(IImfs)
print("*****************")
print(res)