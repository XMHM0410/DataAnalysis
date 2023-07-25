import imp
import pandas as pd
from scipy.fftpack import fft, ifft
import numpy as np
from matplotlib import pyplot as plt
from PyLMD import LMD
from tiqutezheng import get_PFs
def get_PFs(data2,PFs_num):
    """ df分解成活干个PF分量 return PFs, res """
    lmd = LMD(max_num_pf=PFs_num)

    
    # print(y) if debug == True  else None
    PFs, res = lmd.lmd(data2)
    print('PF分量个数：',len(PFs))
    return PFs,res

def yuuchuli(data,sx,xx,pc):
    """ 信号预处理，消除信号偏置，消除随机粗大误差点 """
    for i in range(len(data)):
        if data[i]>sx:
            data[i] = sx + 0.2*(data[i]-sx) - pc
        elif data[i]< xx:
            data[i] = xx - 0.2*(xx-data[i]) - pc
        else:
            data[i] = data[i] - pc
    return data



def pinpuhouchuli(data,yuzhi,):
    """ 频谱后处理 """
    for i in range(len(data)):
        if data[i]>sx:
           pass
    return data



def to_fft(singal,fs):
    fft_y = fft(singal)

    N=len(fft_y)
    x = np.arange(N)           # 频率个数
    # print(x)
    abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
    # angle_y=np.angle(fft_y)              #取复数的角度

    normalization_y=abs_y/N 
    
    half_x = x[range(int(N/2))]*fs/N
    # print(half_x)                                 #取一半区间
    normalization_half_y =2 * normalization_y[range(int(N/2))]      #由于对称性，只取一半区间（单边频谱）
    plt.figure(3)
    plt.plot(half_x[1:],normalization_half_y[1:],'b')
    plt.title('单边频谱(归一化)',fontsize=9,color='blue')
    plt.show()  

if __name__ =='__main__':
    df = pd.read_csv('./3400r.csv')
    length = 1000
    t = np.array(df.loc[1000:length+1000 ,['time']])
    shuju = np.array(df.loc[0:length ,'dl_U'])
    plt.figure(1)
    plt.plot(t,shuju)

    sx = 10
    xx = -10
    pc = 0
    data1 = yuuchuli(shuju,sx,xx,pc)
    plt.figure(2)
    plt.plot(t,data1)
    
    PFs, re = get_PFs(data1,8)

    # fft
    to_fft(data1,2000)
    plt.show()


    
    


