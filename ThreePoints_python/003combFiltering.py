"""
经过梳状滤波器过滤后，50Hz谐波被过滤掉，25Hz保留下来
经过其互补滤波器后，25Hz被过滤，其50Hz被保留。"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%读取滤波后的数据
df = pd.read_csv('ThreePoints_Python\Data\ThreePointsResultData.csv')
x = df['x'].values
