import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('test1.csv')

# 索引第一列数据
x = df.iloc[:, 0]

# 绘制折线图
plt.plot(x)

# 显示图像
plt.show()
