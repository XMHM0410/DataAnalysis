from scipy.signal import argrelextrema
import numpy as np
x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
result = argrelextrema(x, np.greater)
# (array([3, 6]),)
