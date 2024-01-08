import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft

class Generate_Sim_Signal:
    def __init__(self,N,rpm,Fs,p1,p2,p3,k):
        #参数
        self.N = N       # 采样点数
        self.rpm = rpm   # 转速 rpm
        self.Fs = Fs     # 采样率 Hz
        self.p1 = p1     # 传感器1角度 °
        self.p2 = p2     # 传感器1角度 °
        self.p3 = p3     # 传感器1角度 °
        self.k = k       # 偏心角度
    def generate(self):
        n = self.N   
        N = n
        Fs = self.Fs
        dps = self.rpm / 60 * 360 #将转速从r/min转换为°/s
        pps = (N/360) * dps # 每秒采集的点数 
        if pps != Fs :
            self.Fs = pps
            Fs = self.Fs
        t = np.arange(1, n+1)  # 生成1到n的整数序列作为时间点
        xx = ((self.p2 - self.p1)/360)*N #第1、2个传感器相差的点数
        yy = ((self.p3 - self.p2)/360)*N #第2、3个传感器相差的点数
        # 生成圆度轮廓曲线
        r1 = 0.6 * np.cos(30 * 2 * np.pi * t + 0.7 * np.pi) \
            + 3.5 * np.cos(4.22 * 2 * np.pi * t + 0.5 * np.pi) \
            + 1.3 * np.cos(1.7 * 2 * np.pi * t + 0.3 * np.pi) \
            + 3.3 * np.cos(24 * 2 * np.pi * t + 0.1 * np.pi) \
            + 0.9 * np.cos(36 * 2 * np.pi * t + 0.6 * np.pi) #角度需要转化为弧度，因此0.7、0.5、0.3、0.1、0.6要分别乘以np.pi，才能与np.cos()函数的参数对应
        e = 2 + 0.05 * np.random.uniform(-1, 1, size=n)   # 生成长度为n的向量e，其中每个元素都是根据公式计算得到的
        kk = np.random.uniform((k - 0.5) / 180 * np.pi, (k + 0.5) / 180 * np.pi, size=n) #将角度转化为弧度，需要乘以 $\pi/180$。
        r2 = e * np.cos(kk)  # 对e和kk做对应元素的乘积
        r3 = e * np.sin(kk)  # 偏心导致的误差在x,y轴上的分量
        # 构建一个存在两个周期r1的矩阵r，方便对r1进行移相
        r = np.ones(2 * n)  # 生成长度为2*n，元素都为1的向量r
        r[:n] = r1  # 将r1的值赋值给r的前n个元素
        r[n:] = r1  # 将r1的值赋值给r的后n个元素
        return({"r1":r1,"r2":r2,"r3":r3,})

class ThreePointsMethod:
    def __init__(self):
        pass

N = 3600
rpm = 360
Fs = 3600
p1 = 0
p2 = 120
p3 = 240
k = 45
SimSignal = Generate_Sim_Signal(N,rpm,Fs,p1,p2,p3,k)
Signal = SimSignal.generate()

fig, axs = plt.subplots(3, 1, figsize=(10, 8)) 
axs[0].plot(Signal.get("r1"), linewidth = 1, label='圆度误差')
axs[1].plot(Signal.get("r2"), linewidth = 1, label='偏心误差x轴分量')
axs[2].plot(Signal.get("r3"), linewidth = 1, label='偏心误差y轴分量')
plt.show()