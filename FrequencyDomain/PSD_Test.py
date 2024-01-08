import numpy as np
from numpy import sin,cos,pi
import matplotlib.pyplot as plt
import scipy.signal as sig

def PSD(Rx,maxlag,Nfft):
    #PSD Computation of PSD using Autocorrelation Lags
    # (Sx,omega) = PSD(Rx,lags,Nfft)
    # Sx: Computed PSD values
    # omega: digital frequency array in pi units from -1 to 1
    # Rx: autocorrelations from -maxlag to +maxlag
    # maxlag: maximum lag index (must be >= 10)
    # Nfft: FFT size (must be >= 512)
    Nfft2 = Nfft//2
    M = 2*maxlag+1; # Bartlett window length
    Rx = np.bartlett(M)*Rx[len(Rx)//2-maxlag+1:len(Rx)//2+maxlag+2] # Windowed autocorrelations
    Rxzp = np.r_[np.zeros(Nfft2-maxlag),Rx,np.zeros(Nfft2-maxlag-1)]   
    Rxzp = np.fft.ifftshift(Rxzp) #Zero-padding and circular shifting
    Sx = np.fft.fftshift(np.real(np.fft.fft(Rxzp))) # PSD
    Sx = np.r_[Sx,Sx[1]]  # Circular shifting
    omega = np.linspace(-1,1,Nfft+1) # Frequencies in units of pi
    return Sx, omega
# %% 生成一个随机信号x，绘制自相关曲线和功率谱曲线。
x=np.random.rand(1,100); 
maxlag = 10; #Load random sequence data
Rx = sig.correlate(x,x, mode='full')
lags = sig.correlation_lags(x.size, x.size, mode='full')
(Sx,omega) = PSD(Rx[0],maxlag,512); # Compute PSD
plt.subplot(311)
n=np.linspace(0,99,100)
plt.plot(n,x[0])
plt.title('x')
plt.xlabel('n')
plt.subplot(312)
plt.plot(lags,Rx[0])
plt.title('Rxx')
plt.xlabel('lag')
plt.subplot(313)
plt.plot(omega,Sx)
plt.title('Sxx')
plt.xlabel('$w$')
plt.tight_layout()
plt.show()
# plt.savefig('psd.jpg')
