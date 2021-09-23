import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.tensor import Tensor

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
X = torch.tensor(final, requires_grad=True)

threshold = 1
clusterer = TopoGrad(k_kde=30, k_rips=30, scale=0.1, threshold=threshold, n_iter=200, lr=0.01)
destnum = 5
clusterer.destnum = destnum
preds = clusterer(X)
if threshold == 1:
    clusterer.plot()
    plt.show()
print(preds.unique())
print(clusterer.barcodes)


# def _loss(pts: Tensor, plot: bool) -> Tensor:
#     DX = ((pts[None] - pts[:, None]) ** 2).sum(-1)
#     complex = gd.RipsComplex(distance_matrix=DX)
#     st = complex.create_simplex_tree(max_dimension=1)
#     barcodes = st.persistence()
#     # pairs = torch.tensor(barcodes[1], dtype=torch.long)
#     # breakpoint()
#     # p = st.persistence_pairs()
#     # Keep only pairs that contribute to H1, i.e. (edge, triangle), and separate birth (p1b) and death (p1d)
#     # if plot:
#     gd.plot_persistence_diagram(barcodes)
#     plt.show()
#     plt.close()
#     # Total persistence is a special case of Wasserstein distance
#     diag = torch.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], dim=-1)
#     perstot1 = wasserstein_distance(diag1, [], order=1, enable_autodiff=True)
#     plt.show()
#     # p1b = torch.tensor([i[0] for i in p if len(i[0]) == 1])
#     # p1d = torch.tensor([i[1] for i in p if len(i[0]) == 1])
#     # # Same as the finite part of st.persistence_intervals_in_dimension(1), but differentiable
#     # breakpoint()
#     # # diag1 = torch.norm(pts[i1[:, (0, 2)]] - pts[i1[:, (1, 3)]], dim=-1)
#     # # Total persistence is a special case of Wasserstein distance
#     # perstot1 = wasserstein_distance(diag1, [], order=1, enable_autodiff=True)
#     return -perstot1


# # Start with a square, the loss will round the corners
# # opt = torch.optim.SGD([X], lr=0.1)
# # for idx in range(600):
# #     opt.zero_grad()
# #     _loss(X, plot=idx % 100 == 99).backward()
# #     opt.step()
# # Draw every 100 epochs
# # if idx % 100 == 99:

# #     P = X.detach().numpy()
# #     plt.scatter(P[:, 0], P[:, 1])
# #     plt.show()
