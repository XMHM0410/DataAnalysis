import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################陀螺随机漂移误差的Allan分析python代码实现################################

def get_allen(y, tau0=1):
    N = len(y)
    NL = N
    Tau = [] # 保存不同的tau
    Sigma = [] # 保存不同tau下的Allen方差值
    Err = []
    for k in (1,1000):
        sigma_k = np.sqrt(1/2*(NL-1))*np.sum(np.power(y[1:NL]-y[0:NL-1],2)) # Allan的时域表达式
        Sigma.append(sigma_k)
        tau_k = 2**(k-1)*tau0 # 将取样时间加倍，tau2 = 2 tau1
        Tau.append(tau_k)
        err = 1/np.sqrt(2*(NL-1))
        Err.append(err)
        NL = int(np.floor(NL/2))
        if NL < 3:
            break
        y = 1/2*(y[0:2*NL:2]+y[1:2*NL:2]) # 对应序列长度减半
    return Sigma, Tau

data_size_ex = 5
y = np.random.standard_normal((10**data_size_ex),) + (10 ** -data_size_ex) * np.arange(1,10**data_size_ex + 1)
fig, ax = plt.subplots()
Sigma, Tau = get_allen(y)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_adjustable("datalim")
ax.plot(Tau, Sigma)
ax.grid()
plt.show()