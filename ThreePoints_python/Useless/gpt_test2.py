# 首先，频域三点法是一种主轴回转误差分离算法，其步骤如下：
# 1. 将轮廓曲线进行离散化，并通过FFT将其转换为频域信号。
# 2. 对频域信号进行振幅谱和相位谱的分解。
# 3. 根据振幅谱和相位谱计算主轴方向和主轴回转误差。
# 4. 通过主轴回转误差对原始曲线进行旋转校正，得到校正后的曲线。

# 接下来是Python实现该算法的脚本：

# ```python
import numpy as np
from scipy.fft import fft, ifft

# 定义频域三点法函数
def freq_domain_three_point_method(x, y):
    # 将轮廓曲线进行中心化
    x -= np.mean(x)
    y -= np.mean(y)
    # 对离散化后的轮廓曲线进行FFT
    fx = fft(x)
    fy = fft(y)
    # 计算振幅谱和相位谱
    amp_spec = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
    phase_spec = np.arctan2(fy, fx)
    # 计算主轴方向
    theta = 0.5 * np.arctan2(2 * np.sum(amp_spec * np.sin(phase_spec)), np.sum(amp_spec * np.cos(phase_spec)))
    # 对频域信号进行旋转校正
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords = np.vstack([x, y])
    rotated_coords = np.dot(R, coords)
    x_corrected, y_corrected = rotated_coords
    # 计算最小二乘圆
    A = np.vstack([x_corrected, y_corrected, np.ones_like(x_corrected)]).T
    b = x_corrected ** 2 + y_corrected ** 2
    center = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
    xc, yc, r = center
    # 返回极坐标图数据
    theta_grid = np.linspace(0, 2*np.pi, 1000)
    r_grid = r * np.ones_like(theta_grid)
    return theta_grid, r_grid

# 测试脚本
if __name__ == '__main__':
    # 定义一个仿真轮廓曲线
    t = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(t) + 0.1 * np.sin(20*t)
    y = np.sin(t) + 0.1 * np.sin(10*t)
    # 调用频域三点法函数
    theta, r = freq_domain_three_point_method(x, y)
    # 绘制极坐标图
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r)
    ax.set_rlim([0, 1.5*np.max(r)])
    plt.show()
# ```

# 该脚本接受两个一维NumPy数组x和y，表示输入的仿真轮廓曲线。函数首先将轮廓曲线进行中心化，然后使用FFT将其转换为频域信号。接着计算振幅谱和相位谱，并根据振幅谱和相位谱计算主轴方向和主轴回转误差。最后对频域信号进行旋转校正，并计算出最小二乘圆，并返回极坐标图数据。

# 在测试脚本中，我们定义了一个仿真轮廓曲线，并调用freq_domain_three_point_method函数进行分析。然后使用Matplotlib库绘制出极坐标图，并展示出来。

# 报错：phase_spec = np.arctan2(fy, fx)
# TypeError: ufunc 'arctan2' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''