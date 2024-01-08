import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Allan方差计算
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