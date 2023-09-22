from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%读取同步异步误差
df = pd.read_csv('ThreePoints_Python\Data\ThreePointsResultData.csv')
sync = df['Sync'].values
Async = df['Async'].values
# 'Sync': abs(Sync_sig), 'Async': abs(Async_sig), 'Rod': abs(Rod_sig), 'Pos': abs(Pos_sig)