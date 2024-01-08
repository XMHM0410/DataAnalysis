import numpy as np
from scipy.spatial.distance import pdist
# 兰氏距离
# x = np.random.random(5)
x = np.array([0.75173729, 0.34763686, 0.71927609, 0.24151473, 0.22294162])
# y = np.random.random(5)
y = np.array([0.98036113, 0.45482745, 0.87472311, 0.92923963, 0.62922737])
xy = np.sum( np.true_divide( np.abs(x - y), np.abs(x) + np.abs(y) ) )
print(xy)
# 1.4272762731136441
# pdist(xy, metric="canberra")
 # array([1.42727627])
