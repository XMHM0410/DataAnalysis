import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
################################ 定义超参数 ################################

input_dim = 1
hidden_dim = 64
output_dim = 1
num_layers = 2
learning_rate = 0.01
num_epochs = 1000
loss_function = nn.MSELoss()

############################# 定义数据集和数据加载器 ########################

# 生成正弦信号叠加随机噪声信号作为数据集 # 导入txt中的源信号第一列作为训练集
t = np.arange(0, 10, 0.001)
data = pd.read_csv('originaldata.txt', header=None, delimiter='\t')
signal = data.iloc[:, 1].values
# signal = np.sin(t) + 0.1 * np.random.randn(len(t))

# 划分训练集和测试集
train_size = 8000
train_signal = signal[:train_size]
test_signal = signal[train_size:]

# 数据预处理，将信号划分为若干个时间窗口，并使用重叠方式进行采样
window_size = 32
overlap_size = 16

def create_dataset(signal):
    data = []
    for i in range(0, len(signal)-window_size+1, overlap_size):
        window = signal[i:i+window_size]
        data.append(window)
    return np.array(data)

train_data = create_dataset(train_signal)
test_data = create_dataset(test_signal)

# 将数据转换为PyTorch张量，并创建数据加载器
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

######################## 定义神经网络模型 #########################

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, _ = self.gru(out)
        out = self.fc(out[:, -1, :])
        return out

########################## 定义损失函数和优化器 ######################

net = Net(input_dim, hidden_dim, output_dim, num_layers)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

######################### 训练模型 ###################################

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        inputs = torch.tensor(data.reshape(-1, window_size, input_dim))
        labels = torch.tensor(data.reshape(-1, window_size, output_dim)[:,-1,:])

        optimizer.zero_grad()
        outputs = net(inputs.float())
        loss = loss_function(outputs, labels.float())
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

############################# 测试模型并保存 ##############################

# 储存训练好的模型以备使用
torch.save(net.state_dict(), 'model.pth')

# 读取保存的模型
# net = Net(input_dim, hidden_dim, output_dim, num_layers)
# net.load_state_dict(torch.load('model.pth'))

# 使用测试集对模型进行评估
net.eval()
with torch.no_grad():
    test_inputs = torch.tensor(test_data.reshape(-1, window_size, input_dim))
    predicted = net(test_inputs.float()).numpy()

# 将输出结果转换回原始信号的格式
filtered_signal = np.zeros_like(signal)
for i in range(len(predicted)):
    filtered_signal[train_size+i*overlap_size:train_size+(i+1)*overlap_size] = predicted[i,0]

# 绘制原始信号和滤波后的信号
plt.plot(t, signal, label='Original Signal')
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.legend()
plt.show()
