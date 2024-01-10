import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CanberraDistance import CanberraDistance
# %%读文件中的imf1-11
df = pd.read_csv('ThreePoints_EMD_Improve\Data\IMFs.csv')
S = df["x"].values
# S = df["S"].values
t = df["t"].values
imf1 = df["IMF1"].values
imf2 = df["IMF2"].values
imf3 = df["IMF3"].values
imf4 = df["IMF4"].values
imf5 = df["IMF5"].values
imf6 = df["IMF6"].values
imf7 = df["IMF7"].values
imf8 = df["IMF8"].values
imf9 = df["IMF9"].values
imf10 = df["IMF10"].values
imf11 = df["IMF11"].values
# %%计算imf1-11的兰氏距离
cd1 = CanberraDistance.canberry_distance(imf1, S)
cd2 = CanberraDistance.canberry_distance(imf2, S)
cd3 = CanberraDistance.canberry_distance(imf3, S)
cd4 = CanberraDistance.canberry_distance(imf4, S)
cd5 = CanberraDistance.canberry_distance(imf5, S)
cd6 = CanberraDistance.canberry_distance(imf6, S)
cd7 = CanberraDistance.canberry_distance(imf7, S)
cd8 = CanberraDistance.canberry_distance(imf8, S)
cd9 = CanberraDistance.canberry_distance(imf9, S)
cd10 = CanberraDistance.canberry_distance(imf10, S)
cd11 = CanberraDistance.canberry_distance(imf11, S)
# %%输出文件
out = pd.DataFrame({'cd1': cd1,
                    'cd2': cd2, 
                    'cd3': cd3, 
                    'cd4': cd4, 
                    'cd5': cd5, 
                    'cd6': cd6, 
                    'cd7': cd7, 
                    'cd8': cd8, 
                    'cd9': cd9, 
                    'cd10': cd10, 
                    'cd11': cd11})
out.to_csv('ThreePoints_EMD_Improve\Data\CanberraDistance.csv', index=False)