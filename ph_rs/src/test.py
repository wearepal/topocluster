import numpy as np
from rust_ext import merge_h0


def test():
    graph = np.array([[0, 2], [3, 1], [2, 1], [0, 1]])
    dmap = np.array([3.0, 2.0, 0.0, 1.0])
    labels = merge_h0(graph, dmap, 0.0)
    print(labels)

test()
