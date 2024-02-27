import numpy as np

a = np.arange(12).reshape((4, 3))
print(a)
idx = np.random.choice(len(a), 2, replace=False)
print('idx = ', idx)
print(a[idx])