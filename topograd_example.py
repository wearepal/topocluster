import time

import matplotlib.pyplot as plt
import numpy as np

from topocluster.optimisation.unsupervised import TopoCluster

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

X = final

clusterer = TopoCluster(k_rips=30, k_kde=10, scale=0.1, umap_kwargs=None, batch_size=None)
cluster_labels, lifespans = clusterer.fit(X, threshold=1)

import time

start = time.time()

repeats = 1000
for i in range(repeats):
    cluster_labels, lifespans = clusterer.fit(X, threshold=1)

avg_time = (time.time() - start) / repeats
print(f"Average execution time over {repeats} repeats: {avg_time}")

# dddd is cluster label, but when you set thresh to be 1 , ALL the candidate clusters are merged, for this dataset there is only one cluster
# left after merging. pd is the overall persistence diagram. For this dataset we can see there is only 1 salient candidate cluster. If we set
# the thresh between the persistence of most salient and the rest. We can see 2 cluster shown in next. To find this thresh we just need to use
# birth time minus death time to get the persistence time of candidate clusters.
# we can see 0.35791 is so much bigger than the rest (0.0744). So if we set a number between 0.35791 and 0.0744. The corresponding candidate
# cluster will be kept.

# pd_plot = clusterer.plot_pd(lifespans)

# res, pd = topograd(X, 30, 10,0.1,2,0.01,500)
