import numpy as np
import matplotlib.pyplot as plt
"""
Allan方差是一种用于分析时间序列数据的方法，可以用于确定数据的稳定性和噪声特性。
如果Allan方差曲线呈现出平稳的趋势，说明数据在不同的时间间隔内的噪声特性比较一致，因此可以认为该信号稳定性好，噪声较低。
但是需要注意的是，Allan方差曲线只能反映数据的噪声特性，不能反映数据的准确性。
因此，在实际应用中，需要综合考虑数据的准确性和稳定性，以便更好地评估数据的质量。
"""
def allan_variance(data, tau = 10):
    """
    计算Allan方差
    :param data: 数据数组
    :param tau: 时间间隔 默认取10
    :return: Allan方差数组
    """
    n = len(data)
    max_m = int(np.floor(n / (2 * tau)))
    if max_m == 0:
        raise ValueError("tau is too large for the given data array")
    allan_var = np.zeros(max_m)
    for m in range(1, max_m + 1):
        sum_var = 0
        for i in range(1, n - 2 * m * tau + 1):
            denominator = 2 * (n - 2 * m * tau) * tau ** 2
            if denominator == 0:
                raise ValueError("denominator is zero")
            sum_var += (data[i + 2 * m * tau - 1] - 2 * data[i + m * tau - 1] + data[i - 1]) ** 2 / denominator
        allan_var[m - 1] = sum_var
    return allan_var

data = np.random.randn(1000)
tau = 10
allan_var = allan_variance(data, tau)
print(allan_var)
# 绘制Allan方差曲线
"""
通常可以使用图形化的方式来表示Allan方差数组。
常见的图形化表示方法是Allan方差曲线，其中横轴表示时间间隔，纵轴表示Allan方差的值。
通过绘制Allan方差曲线，可以更直观地观察数据的稳定性和噪声特性。
例如，如果Allan方差曲线呈现出明显的趋势或周期性变化，则说明数据存在明显的噪声或漂移。
相反，如果Allan方差曲线呈现出平稳的趋势，则说明数据比较稳定。
"""
plt.loglog(np.arange(1, len(allan_var) + 1) * tau, allan_var)
plt.xlabel('Time Interval (s)')
plt.ylabel('Allan Variance')
plt.title('Allan Variance Curve')
plt.show()