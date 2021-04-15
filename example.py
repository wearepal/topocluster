import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from topocluster.clustering.topograd import TopoGrad

k_kde = 30
k_rips = 30
scale = 0.1
destnum = 2

torch.use_deterministic_algorithms(True)
np.random.seed(0)
mean = [3.1, 0]
cov = [[0.07, 0], [0, 0.07]]
x, y = np.random.multivariate_normal(mean, cov, 500).T

mean = [0, 0]
cov = [[0.7, 0], [0, 0.7]]
x1, y2 = np.random.multivariate_normal(mean, cov, 500).T

finalx = np.append(x1, x)
finaly = np.append(y2, y)
final = np.zeros([1000, 2])
final[:, 0] = finalx
final[:, 1] = finaly

mean = [0, 0]
cov = [[0.7, 0], [0, 0.7]]

pc_np = final
pc = torch.tensor(pc_np, requires_grad=True)
cl = TopoGrad(k_kde=30, k_rips=30, scale=0.1, n_iter=100, threshold=1, lr=0.01 )
cl.destnum = 5
cl.bias = nn.Parameter(torch.randn(pc.shape), requires_grad=True)
cl(pc, threshold=1)
cl.plot()
print(cl.bias)
plt.show()
# print(cl(pc, threshold=0.8))
