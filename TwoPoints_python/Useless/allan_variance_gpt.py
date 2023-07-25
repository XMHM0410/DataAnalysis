import numpy as np
import pandas as pd

def allan_variance(signal, fs):
    tau = np.logspace(-3, np.log10(len(signal)/2/fs), 100)
    print(tau)
    # tau = tau.astype(int)
    av = np.zeros(len(tau))
    for i in range(len(tau)):
        n = int(np.floor(len(signal)/tau[i]))
        x = signal[:n*np.floor(tau[i]).astype(int)].reshape(n, np.floor(tau[i]).astype(int))
        y = np.mean(x, axis=1)
        av[i] = np.mean((y[1:]-y[:-1])**2)/2
    return av

def main_error_coefficient(signal, fs):
    tau = np.logspace(-3, np.log10(len(signal)/2/fs), 100)
    av = allan_variance(signal, fs)
    slope, intercept = np.polyfit(np.log10(tau), np.log10(av), 1)
    return slope/2

data = pd.read_csv('data.csv')
signal = data['signal'].values
filtered_signal = data['filtered_signal'].values

fs = 1000  # 假设采样率为1000Hz
signal_error = main_error_coefficient(signal, fs)
filtered_signal_error = main_error_coefficient(filtered_signal, fs)

print('Signal random error coefficient:', signal_error)
print('Filtered signal random error coefficient:', filtered_signal_error)