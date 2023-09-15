'''
Author: Dianye Huang
Date: 2023-01-14 10:26:47
LastEditors: Dianye Huang
LastEditTime: 2023-01-14 14:37:02
Description: 
'''
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pandas as pd

class myFFT:
    def __init__(self):
        pass
    
    def get_spetrum(self, data, fs, flag_plt=False):
        N = len(data)
        fft_y = fft(data)        # get complex number
        abs_y = np.abs(fft_y)    # get magnitude
        ang_y = np.angle(fft_y)  # get phase
        nrm_y = abs_y/(N/2)      # get normailzed magnitude (A0/N, A1.../(N/2))
        nrm_y[0] = nrm_y[0]/2      
        nmh_y = nrm_y[:int(N/2)] # half normalized magnitude
        agh_y = ang_y[:int(N/2)] # half normalized angle
        spes  = np.arange(int(N/2))*fs/N  # specturm axis 
        
        if flag_plt:
            plt.figure()
            x = np.arange(N)
            plt.subplot(2,3,1)
            plt.plot(x/fs, data)
            plt.title('raw signal')
            plt.xlabel('Time (s)')
            
            plt.subplot(2,3,4)
            plt.plot(x[:50]/fs, data[:50])
            plt.title('partial raw signal')
            plt.xlabel('Time (s)')
            
            plt.subplot(2,3,2)
            plt.plot(x, abs_y)
            plt.title('magnitudes')
            plt.xlabel('Sample index')
            
            plt.subplot(2,3,5)
            plt.plot(x, ang_y)
            plt.title('angles')
            plt.xlabel('Sample index')
            
            plt.subplot(2,3,3)
            plt.plot(x, nrm_y)
            plt.title('nomalized magnitude')
            plt.xlabel('Sample index')
            
            plt.subplot(2,3,6)
            plt.plot(spes, nmh_y)
            plt.title('half nomalized magnitude')
            plt.xlabel('Frequency (Hz)')
            
            plt.show()
        
        return nmh_y, agh_y, spes

    def get_fft(self, data, fs):
        N = len(data)
        fft_y = fft(data)
        tmp_arr = np.arange(0, fs/2, fs/N)
        freq_y = np.hstack((tmp_arr, np.flip(tmp_arr, axis=0)))
        # 这行代码的作用是将一个一维数组tmp_arr进行水平拼接，拼接的内容是tmp_arr数组的翻转。
        # 具体来说，np.flip(tmp_arr, axis=0)表示将tmp_arr数组沿着第0个轴（即行）进行翻转，得到一个新的数组。
        # 然后，使用np.hstack函数将tmp_arr和翻转后的数组进行水平拼接，得到一个新的一维数组freq_y。
        # 这个新数组的长度是原数组的两倍，其中前一半是原数组的内容，后一半是原数组内容的翻转。
        # 这个操作通常用于生成一个对称的频谱，以便进行傅里叶变换。
        return fft_y, freq_y
    
    def create_demo_signal(self, N=2800, fs=1400):
        # create a signal along with sample rate
        t = np.arange(0, N/fs, 1/fs)
        y = 3 + 7*np.sin(2*np.pi*200*t) + 5*np.sin(2*np.pi*400*t) + 3*np.sin(2*np.pi*600*t)
        noise_arr = np.random.normal(0, 2, N)
        return t, y+noise_arr, fs
    
# 读取CSV文件
df = pd.read_csv('freq.csv')

# 获取频率及其对应的幅值
frequencies = df['Frequency'].values
amplitudes = df['Amplitude'].values

mfft=myFFT()
t, data, fs = mfft.create_demo_signal(N=4096, fs=2048)
# mags, angs, spes = mfft.get_spetrum(data, fs, flag_plt=True)
# print(mags)
# print(angs)
# print(spes)

fft_y, freq_y = mfft.get_fft(data, fs)
fft_y[freq_y>450] = 0 # 简单的滤波，除去大于450Hz的分量利用ifft重新组合信号的结果
sig = ifft(fft_y).real

print(len(fft_y))
print(len(freq_y))
print(sig)

plt.figure()
plt.plot(t[:50], data[:50])
plt.plot(t[:50], sig[:50])
plt.show()
