import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# a = torch.arange(24).reshape(2, 4, 3)
# b = torch.arange(12).reshape(1, 4, 3)
# print(torch.cat([a, b, torch.empty(0)], dim=0).shape)
# c = a.shape[1:]
# d = np.random.rand(*c)
# print(*a.shape)
# print(d)

# mu = 0.95
# LT_ratio = np.array([mu ** i for i in range(200)]) * 100000
# np.savetxt('res/csv/long_tail.csv', LT_ratio, delimiter=',', fmt="%.6f")

# loss = nn.CrossEntropyLoss()
# x = torch.tensor([[2,5,6]], dtype=torch.float32)
# y = torch.tensor([1])

# alpha = torch.exp(x)
# S = alpha.sum(dim=1, keepdim=True)
# print(loss(x, y))
# print(F.nll_loss(torch.log(alpha) - torch.log(S), y, reduction='none'))

# from torch.distributions import normal
# mu = torch.rand(2)
# print(mu[:1])
# sigma = torch.tensor([4.0, 5.0])
# sampler = normal.Normal(mu, sigma)
# print(sampler.sample([1]).shape)

output_exp = torch.exp(torch.rand(2,3))
max_val, max_idx = torch.max(output_exp, dim=-1)
row_idx = torch.arange(output_exp.shape[0])
print(output_exp)
output_exp[row_idx, max_idx] -= max_val
print(output_exp)