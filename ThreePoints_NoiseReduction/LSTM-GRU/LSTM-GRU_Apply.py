import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('originaldata.txt', header=None, delimiter='\t')
signal = data.iloc[:, 1].values
t = np.arange(0, 10, 0.001)
# signal = 8+np.sin(t) + 0.1 * np.random.randn(len(t))

input_dim = 1
hidden_dim = 64
output_dim = 1
num_layers = 2
learning_rate = 0.01
num_epochs = 1000
loss_function = nn.MSELoss()

window_size = 32
overlap_size = 16

def create_dataset(signal):
    data = []
    for i in range(0, len(signal)-window_size+1, overlap_size):
        window = signal[i:i+window_size]
        data.append(window)
    return np.array(data)

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

train_size = 0
test_signal = signal[train_size:]
# test_signal = signal
test_data = create_dataset(test_signal)

# 读取保存的模型
net = Net(input_dim, hidden_dim, output_dim, num_layers)
net.load_state_dict(torch.load('model.pth'))

# 使用测试集对模型进行评估
net.eval()
with torch.no_grad():
    test_inputs = torch.tensor(test_data.reshape(-1, window_size, input_dim))
    predicted = net(test_inputs.float()).numpy()

# 将输出结果转换回原始信号的格式
filtered_signal = np.zeros_like(signal)

for i in range(len(predicted)):
    filtered_signal[train_size+i*overlap_size:train_size+(i+1)*overlap_size] = predicted[i,0]
    # filtered_signal = predicted[i,0]
signal = signal.reshape(-1, 1)
filtered_signal = filtered_signal.reshape(-1, 1)
data = np.concatenate((signal, filtered_signal), axis=1)
df = pd.DataFrame(data, columns=['signal', 'filtered_signal'])
df.to_csv('data.csv', index=False)
# 绘制原始信号和滤波后的信号
plt.plot(t, signal, label='Original Signal')
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.legend()
plt.show()