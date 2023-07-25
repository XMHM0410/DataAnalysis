# 这个错误通常是由于输入数组的数据类型不兼容导致的。在这里，fy和fx应该是复数数组，因为它们是FFT的输出结果，而arctan2函数只支持实数数组。因此，可以尝试将fy和fx转换为实数数组，例如通过取其绝对值来获取振幅谱。

# 修改代码如下：

# ```python
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
# 定义频域三点法函数
def freq_domain_three_point_method(x, y):
    # 将轮廓曲线进行中心化
    x -= np.mean(x)
    y -= np.mean(y)
    # 对离散化后的轮廓曲线进行FFT
    fx = fft(x)
    fy = fft(y)
    # 计算振幅谱和相位谱
    amp_spec = np.abs(fx + 1j*fy) # 获取振幅谱
    phase_spec = np.angle(fx + 1j*fy) # 获取相位谱
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
    center = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b)) #将计算出的主轴方向应用于旋转矩阵R，并将原始曲线坐标点与该矩阵相乘以得到校正后的坐标点。
    xc, yc, r = center #在第3步计算最小二乘圆的过程中，我们通过求解线性方程组来计算最小二乘圆的圆心坐标和半径。
    #该线性方程组的系数矩阵A的形状为(N, 3)，其中N是离散化后的轮廓曲线上点的个数，3表示三个参数a, b, c，它们构成了最小二乘圆的方程x^2 + y^2 + ax + by + c = 0。而b数组的形状为(N, )，表示最小二乘圆方程右侧的常数项。
    # 因此，在求解线性方程组时，我们得到了包含最小二乘圆圆心坐标xc和yc以及半径r的向量center。
    # 在这一行代码中，我们将center向量的三个元素分别赋值给xc、yc和r变量，以便在后续的代码中使用它们。
    # 由于xc和yc在本脚本中没有被使用，因此看起来它们似乎没有被利用。
    # 但是，实际上它们在计算校正后的曲线时被用作旋转中心。具体来说，在对原始曲线进行旋转校正时，我们需要将原始曲线中所有点都围绕着圆心点(xc, yc)旋转一个角度。因此，在该脚本中，我们将center向量的前两个元素作为旋转中心的坐标点。
    # 返回极坐标图数据
    theta_grid = np.linspace(0, 2*np.pi, 1000)
    r_grid = r * np.ones_like(theta_grid)
    return theta_grid, r_grid

# 测试脚本
# if __name__ == '__main__':
#     # 定义一个仿真轮廓曲线
#     t = np.linspace(0, 2*np.pi, 1000)
#     x = np.cos(t) + 0.1 * np.sin(20*t)
#     y = np.sin(t) + 0.1 * np.sin(10*t)
#     # 调用频域三点法函数
#     theta, r = freq_domain_three_point_method(x, y)
#     # 绘制极坐标图
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#     ax.plot(theta, r)
#     ax.set_rlim([0, 1.5*np.max(r)])
#     plt.show()
# # ```

# 在修改后的代码中，我们使用绝对值函数np.abs和相位函数np.angle来分别获取振幅谱和相位谱。然后再将其传递给主轴方向计算和旋转校正部分的代码进行进一步处理。

# 修改一下代码，在测试脚本中，我需要将仿真轮廓曲线绘制极坐标图作为图1，fft后的振幅谱绘图作为图2，相位谱作为图3，最后结果绘制极坐标图作为图4

# 要将仿真轮廓曲线、振幅谱和相位谱绘制成图像，可以使用Matplotlib库中的plot函数。具体来说，在脚本中添加以下代码可以实现这一功能：

# ```python
if __name__ == '__main__':
    # 定义一个仿真轮廓曲线
    t = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(t) + 0.1 * np.sin(20*t)
    y = np.sin(t) + 0.1 * np.sin(10*t)

    # 计算频域三点法并绘制图像
    theta, r = freq_domain_three_point_method(x, y)
    
    fig, ax4 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax4.plot(theta, r)
    ax4.set_rlim([0, 1.5*np.max(r)])

    fig, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.plot(t, x, label='x')
    ax1.plot(t, y, label='y')
    ax4.set_rlim([0, 1.5*np.max(r)])
    ax1.legend()
    ax1.set_title('Simulated Profile')

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))   

    fx = fft(x)
    fy = fft(y)
    freq = np.fft.fftfreq(len(x), d=t[1]-t[0])
    axs[0].plot(freq, np.abs(fx), label='|fx|')
    axs[0].plot(freq, np.abs(fy), label='|fy|')
    axs[0].legend()
    axs[0].set_xlim([0, 50])
    axs[0].set_ylim([0, 100])
    axs[0].set_title('Amplitude Spectrum')

    axs[1].plot(freq, np.angle(fx), label='angle(fx)')
    axs[1].plot(freq, np.angle(fy), label='angle(fy)')
    axs[1].legend()
    axs[1].set_xlim([0, 50])
    axs[1].set_ylim([-np.pi, np.pi])
    axs[1].set_title('Phase Spectrum')

    plt.show()
# ```

# 其中，我们首先定义了一个仿真轮廓曲线，并计算频域三点法。然后，使用Matplotlib库中的subplot函数创建一个2*2的子图，并在不同的子图中绘制出仿真轮廓曲线、校正后的曲线、振幅谱和相位谱。最后，使用plt.show()函数展示图像。

# 值得注意的是，由于频域信号通常有复数部分，因此在绘制振幅谱和相位谱时，需要将复数数组转换为实数数组。可以使用np.abs函数获取振幅谱，并使用np.angle函数获取相位谱。此外，FFT输出的频率数组是以采样点数作为单位的，因此我们还需要使用np.fft.fftfreq函数来获取实际的频率。