import pandas as pd
import numpy as np
from PyLMD import LMD

from matplotlib import  pyplot as plt

from scipy.fftpack import fft, ifft

"""LMD;局部均值分解;Local Mean Decomposition"""


def to_fft(singal,fs):
    fft_y = fft(singal)

    N=len(fft_y)
    x = np.arange(N)           # 频率个数
    print(x)
    abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
    angle_y=np.angle(fft_y)              #取复数的角度

    normalization_y=abs_y/N 
    
    half_x = x[range(int(N/2))]*fs/N
    print(half_x)                                 #取一半区间
    normalization_half_y =2 * normalization_y[range(int(N/2))]      #由于对称性，只取一半区间（单边频谱）
    plt.figure(2)
    plt.plot(half_x,normalization_half_y,'b')
    plt.title('单边频谱(归一化)',fontsize=9,color='blue')
    plt.show()



if __name__ == '__main__':
    """原来这段被引掉"""
    t = np.linspace(0,1.99,500)
    x = 2*np.sin(2*np.pi*5*t) + 5*np.sin(2*np.pi*11*t) + 7*np.cos(2*np.pi*25*t)+ 10*np.sin(2*np.pi*60*t)
    to_fft(x,250)
    lmd  = LMD(max_num_pf=8,include_endpoints=True)
    PFs,residue = lmd.lmd(x)
    print(len(PFs))
    subplotNum = len(PFs) + 1
    plt.figure(1)
    for i in range(1, subplotNum-1):
        plt.subplot(subplotNum, 1,i )
        plt.title('PF%d' % i)
        plt.plot(t, PFs[i-1])
    plt.subplot(subplotNum, 1, subplotNum)
    plt.title('residue')
    plt.plot(t, residue)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    to_fft(PFs[4],250)
    """20220306声发射"""
    