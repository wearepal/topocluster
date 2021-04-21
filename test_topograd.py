import matplotlib.pyplot as plt
import numpy as np
import torch

from topocluster.clustering.topograd import TopoGrad

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
X = torch.as_tensor(final)

clusterer = TopoGrad(k_kde=30, k_rips=30, scale=0.1, threshold=1, n_iter=100, lr=0.01)
clusterer.destnum = 4
preds = clusterer(X)
clusterer.plot()
plt.show()
