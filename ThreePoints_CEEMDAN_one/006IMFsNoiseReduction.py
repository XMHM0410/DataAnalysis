import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CanberraDistance import canberraDistance
from NoiseReduction import kalman, wavelet
# %%读文件中的imf1-10
dfi1 = pd.read_csv('ThreePoints_CEEMDAN_one\Data\downSample\MixedSignal6.csv')
S = dfi1["S"].values
theta = dfi1["theta"].values
dfi2 = pd.read_csv('ThreePoints_CEEMDAN_one\Data\IMFs6.csv')
imf1 = dfi2["IMF1"].values
imf2 = dfi2["IMF2"].values
imf3 = dfi2["IMF3"].values
imf4 = dfi2["IMF4"].values
imf5 = dfi2["IMF5"].values
imf6 = dfi2["IMF6"].values
imf7 = dfi2["IMF7"].values
imf8 = dfi2["IMF8"].values
imf9 = dfi2["IMF9"].values
imf10 = dfi2["IMF10"].values
# %%噪声信号卡尔曼滤波 imf123
imf1_NR = kalman.Kalman_denoise(imf1)
print(len(imf1))
print(len(imf1_NR))
imf2_NR = kalman.Kalman_denoise(imf2)
print(len(imf2_NR))
imf3_NR = kalman.Kalman_denoise(imf3)
# %%混叠信号小波阈值降噪 imf456
imf4_NR = wavelet.Wavelet_denoise(imf4)
print(len(imf4_NR))
# 去掉imf4_NR的最后一个
imf4_NR = imf4_NR[:-1]
print(len(imf4_NR))
imf5_NR = wavelet.Wavelet_denoise(imf5)
imf5_NR = imf5_NR[:-1]
imf6_NR = wavelet.Wavelet_denoise(imf6)
imf6_NR = imf6_NR[:-1]
# %%降噪结果导出
out = pd.DataFrame({'IMF1': imf1_NR,
                    'IMF2': imf2_NR, 
                    'IMF3': imf3_NR, 
                    'IMF4': imf4_NR, 
                    'IMF5': imf5_NR, 
                    'IMF6': imf6_NR, 
                    'IMF7': imf7, 
                    'IMF8': imf8, 
                    'IMF9': imf9, 
                    'IMF10': imf10, 
                    })
out.to_csv('ThreePoints_CEEMDAN_one\Data\IMFsNoiseReduction.csv', index=False)