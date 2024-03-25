import numpy as np
import matplotlib.pyplot as plt
import torch

from numpy.random import pareto

# labels =  np.array([857, 576, 771, 577, 785, 711, 848, 632, 487, 602])
# xs = pareto(6, size=sum(labels))
# # plt.hist(xs, bins=25)
# # plt.show()
# hist2, bin_edges2 = np.histogram(xs, bins=20, density=True)
# hist2 = hist2[:10]
# plt.plot(hist2 / sum(hist2))
# plt.show()
# print('hello')
# print(hist2 * max(labels))
a = torch.arange(24).reshape(2, 4, 3)
b = torch.arange(12).reshape(1, 4, 3)
print(torch.cat([a, b, torch.empty(0)], dim=0).shape)
print(a[:, 0].shape)