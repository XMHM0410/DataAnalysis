import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv(r'C:\Users\lixim\Desktop\LoggedData.csv')

# 获取第一列数据
data = df.iloc[:, 0]

# 绘制图像
plt.plot(data)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('First Column Data')
plt.show()
