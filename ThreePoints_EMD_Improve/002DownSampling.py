import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %% 测量原始数据
df = pd.read_csv('ThreePoints_EMD_Improve\Data\OriginalSignal.csv')
s1 = df["filtered_s1"].values
s2 = df["filtered_s2"].values
s3 = df["filtered_s3"].values
# %%将相邻两个采样点取平均降低采样率
def downsample2(data):
    odd = data[1::2]
    even = data[0::2]
    if odd.shape[0] != even.shape[0]:
        even = data[0::2][:-1] # 偶数向去掉最后一项
        downsampled_data = (odd + even) / 2
    else:
        downsampled_data = (odd + even) / 2
    return downsampled_data
def downsample_loop(data):  
    for i in range(12):  # 12次循环，因为最后一次的输出不需要再作为下一次的输入  
        processed_data = downsample2(data)  # 假设downsample()返回处理后的数据  
        data = processed_data  # 将处理后的数据作为下一次的输入  
    return data  # 返回最后一次处理后的数据
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
# %%第一次降采样率400000
s1_ds1 = downsample2(s1)
s2_ds1 = downsample2(s2)
s3_ds1 = downsample2(s3)
plotds(s1_ds1,s2_ds1,s3_ds1)
# %%降13次降到100个点
s1_ds2 = downsample2(s1_ds1)
s2_ds2 = downsample2(s2_ds1)
s3_ds2 = downsample2(s3_ds1)
plotds(s1_ds2,s2_ds2,s3_ds2)
s1_ds3 = downsample2(s1_ds2)
s2_ds3 = downsample2(s2_ds2)
s3_ds3 = downsample2(s3_ds2)
plotds(s1_ds3,s2_ds3,s3_ds3)
s1_ds4 = downsample2(s1_ds3)
s2_ds4 = downsample2(s2_ds3)
s3_ds4 = downsample2(s3_ds3)
plotds(s1_ds4,s2_ds4,s3_ds4)
s1_ds5 = downsample2(s1_ds4)
s2_ds5 = downsample2(s2_ds4)
s3_ds5 = downsample2(s3_ds4)
plotds(s1_ds5,s2_ds5,s3_ds5)
s1_ds6 = downsample2(s1_ds5)
s2_ds6 = downsample2(s2_ds5)
s3_ds6 = downsample2(s3_ds5)
plotds(s1_ds6,s2_ds6,s3_ds6)
s1_ds7 = downsample2(s1_ds6)
s2_ds7 = downsample2(s2_ds6)
s3_ds7 = downsample2(s3_ds6)
plotds(s1_ds7,s2_ds7,s3_ds7)
s1_ds8 = downsample2(s1_ds7)
s2_ds8 = downsample2(s2_ds7)
s3_ds8 = downsample2(s3_ds7)
plotds(s1_ds8,s2_ds8,s3_ds8)
s1_ds9 = downsample2(s1_ds8)
s2_ds9 = downsample2(s2_ds8)
s3_ds9 = downsample2(s3_ds8)
plotds(s1_ds9,s2_ds9,s3_ds9)
s1_ds10 = downsample2(s1_ds9)
s2_ds10 = downsample2(s2_ds9)
s3_ds10 = downsample2(s3_ds9)
plotds(s1_ds10,s2_ds10,s3_ds10)
s1_ds11 = downsample2(s1_ds10)
s2_ds11 = downsample2(s2_ds10)
s3_ds11 = downsample2(s3_ds10)
plotds(s1_ds11,s2_ds11,s3_ds11)
s1_ds12 = downsample2(s1_ds11)
s2_ds12 = downsample2(s2_ds11)
s3_ds12 = downsample2(s3_ds11)
plotds(s1_ds12,s2_ds12,s3_ds12)
s1_ds13 = downsample2(s1_ds12)
s2_ds13 = downsample2(s2_ds12)
s3_ds13 = downsample2(s3_ds12)
plotds(s1_ds13,s2_ds13,s3_ds13)
plt.show()
# %%输出文件
df1 = pd.DataFrame({"s1_ds1":s1_ds1,"s2_ds1":s2_ds1,"s3_ds1":s3_ds1})
df1.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal1.csv', index=False)
df2 = pd.DataFrame({"s1_ds2":s1_ds2,"s2_ds2":s2_ds2,"s3_ds2":s3_ds2})
df2.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal2.csv', index=False)
df3 = pd.DataFrame({"s1_ds3":s1_ds3,"s2_ds3":s2_ds3,"s3_ds3":s3_ds3})
df3.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal3.csv', index=False)
df4 = pd.DataFrame({"s1_ds4":s1_ds4,"s2_ds4":s2_ds4,"s3_ds4":s3_ds4})
df4.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal4.csv', index=False)
df5 = pd.DataFrame({"s1_ds5":s1_ds5,"s2_ds5":s2_ds5,"s3_ds5":s3_ds5})
df5.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal5.csv', index=False)
df6 = pd.DataFrame({"s1_ds6":s1_ds6,"s2_ds6":s2_ds6,"s3_ds6":s3_ds6})
df6.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal6.csv', index=False)
df7 = pd.DataFrame({"s1_ds7":s1_ds7,"s2_ds7":s2_ds7,"s3_ds7":s3_ds7})
df7.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal7.csv', index=False)
df8 = pd.DataFrame({"s1_ds8":s1_ds8,"s2_ds8":s2_ds8,"s3_ds8":s3_ds8})
df8.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal8.csv', index=False)
df9 = pd.DataFrame({"s1_ds9":s1_ds9,"s2_ds9":s2_ds9,"s3_ds9":s3_ds9})
df9.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal9.csv', index=False)
df10 = pd.DataFrame({"s1_ds10":s1_ds10,"s2_ds10":s2_ds10,"s3_ds10":s3_ds10})
df10.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal10.csv', index=False)
df11 = pd.DataFrame({"s1_ds11":s1_ds11,"s2_ds11":s2_ds11,"s3_ds11":s3_ds11})
df11.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal11.csv', index=False)
df12 = pd.DataFrame({"s1_ds12":s1_ds12,"s2_ds12":s2_ds12,"s3_ds12":s3_ds12})
df12.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal12.csv', index=False)
df13 = pd.DataFrame({"s1_ds13":s1_ds13,"s2_ds13":s2_ds13,"s3_ds13":s3_ds13})
df13.to_csv('ThreePoints_EMD_Improve\Data\downSample\DownSapleSignal13.csv', index=False)