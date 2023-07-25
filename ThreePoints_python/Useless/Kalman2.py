import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class KalmanFilter:
    def __init__(self, initial_state_mean, initial_state_covariance, transition_matrix, observation_matrix, process_noise_covariance, observation_noise_covariance):
        """
        初始化卡尔曼滤波器

        参数:
            initial_state_mean (float): 初始状态均值
            initial_state_covariance (float): 初始状态协方差
            transition_matrix (ndarray): 状态转移矩阵
            observation_matrix (ndarray): 观测矩阵
            process_noise_covariance (float): 过程噪声协方差
            observation_noise_covariance (float): 观测噪声协方差
        """
        self.state_mean = initial_state_mean
        self.state_covariance = initial_state_covariance
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance

    def predict(self):
        """
        预测下一时刻的状态
        """
        self.state_mean = np.dot(self.transition_matrix, self.state_mean)
        self.state_covariance = np.dot(np.dot(self.transition_matrix, self.state_covariance), self.transition_matrix.T) + self.process_noise_covariance

    def update(self, observation):
        """
        根据观测值更新状态估计

        参数:
            observation (float): 观测值
        """
        kalman_gain = np.dot(np.dot(self.state_covariance, self.observation_matrix.T), np.linalg.inv(np.dot(np.dot(self.observation_matrix, self.state_covariance), self.observation_matrix.T) + self.observation_noise_covariance))
        self.state_mean = self.state_mean + np.dot(kalman_gain, observation - np.dot(self.observation_matrix, self.state_mean))
        self.state_covariance = np.dot(np.identity(self.state_mean.shape[0]) - np.dot(kalman_gain, self.observation_matrix), self.state_covariance)
    
    def run(self, observations):
        """
        运行卡尔曼滤波器

        参数:
            observations (ndarray): 观测值序列

        返回:
            state_means (ndarray): 估计的状态均值序列
        """
        state_means = []
        for observation in observations:
            self.predict()
            self.update(observation)
            state_means.append(self.state_mean)
        return np.array(state_means)

def save_file(arr1,arr2,file_name):
    # 两个一维数组
    arr1 = pd.Series(arr1)
    arr2 = pd.Series(arr2)

    # 使用 concat 函数将两个数组组合成一个 DataFrame
    df = pd.concat([arr1, arr2], axis=1)

    # 将 DataFrame 输出到 CSV 文件
    df.to_csv(file_name+'.csv', index=False)
    f = pd.concat([arr1, arr2], axis=1)

# 定义状态转移矩阵
transition_matrix = np.array([[1]])
# 定义观测矩阵
observation_matrix = np.array([[1]])
# 定义初始状态均值
initial_state_mean = np.array([0])
# 定义初始状态协方差
initial_state_covariance = np.array([[1]])
# 定义过程噪声协方差
process_noise_covariance = 0.0001
# 定义观测噪声协方差
observation_noise_covariance = 0.1

# 创建卡尔曼滤波器对象
kalman_filter = KalmanFilter(initial_state_mean, initial_state_covariance, transition_matrix, observation_matrix, process_noise_covariance, observation_noise_covariance)
# 定义一个仿真轮廓曲线
t = np.linspace(0, 2*np.pi, 3600)
x = np.cos(t) + 0.1 * np.sin(20*t) + 0.01 * np.cos(10*t)+ 0.001 * np.sin(t)
plt.plot(x, label='Simulation')

#run
state_means=kalman_filter.run(observations=x)
plt.plot(state_means, label='Kalman filtered')

print(x)
print(state_means.T.flatten())
save_file(x,state_means.T.flatten(),"test1")

plt.legend()
plt.show()
