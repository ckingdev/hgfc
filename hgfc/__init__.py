# coding=utf-8
import math

import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing


def divergence(X, Y):
    # Note, this is equivalent to the KL divergence plus the sum of Y minus the sum of X
    # Y may not have *any* nonzero entries, else this is undefined.
    #
    # X = W, matrix of input weights
    # Y = HLH^T, the factorization
    #
    # Since X and Y are going to be sparse, and zero entries for X result in a sum of zero,
    # we can iterate over X's nnz.
    result = Y.sum() - X.sum()
    div = X / Y
    for (i, j) in zip(*X.nonzero()):
        result += X[i, j] * math.log(div[i, j])
    return result


# np.dot(x, y) does matrix multiplication
# using * with numpy matrices does elementwise multiplication
# a * b (scipy) == a.dot(b) (numpy)

def update(W, H, L):
    denom = H * L * H.T
    W_ = W / denom

    # TODO: Can we eliminate this?
    W_[np.isinf(W_)] = 1.0  # 0/0=1
    W_[np.isnan(W_)] = 0.0

    H_ = H.multiply(W_ * H * L)
    H_ = preprocessing.normalize(H_, norm="l1", axis=0)

    L_ = L.multiply(H.T * W_ * H)
    L_ = L_.multiply(W.sum() / L_.sum())

    return H_, L_


def cluster(W, n_clusters):
    H = sp.csr_matrix(
        np.random.random((W.shape[0], n_clusters)))  # TODO: does it make sense to have this sparse?
    H = preprocessing.normalize(H, norm="l1", axis=0)

    L = sp.dia_matrix((H.sum(axis=0), [0]), shape=(H.shape[1], H.shape[1]))

    for i in range(50):  # TODO: find a better stopping criterion
        # HLHT = H * L * H.T
        H, L = update(W, H, L)
    return H, L

def cluster_figure1a():
    np.set_printoptions(suppress=True)
    # This comes from "Soft Clustering on Graphs," as a sanity check.
    W = sp.csr_matrix([[0, 1, 1, 0, 0, 1, 1],
                       [1, 0, 1, 1, 0, 0, 0],
                       [1, 1, 0, 1, 0, 0, 0],
                       [0, 1, 1, 0, 1, 1, 0],
                       [0, 0, 0, 1, 0, 1, 0],
                       [1, 0, 0, 0, 1, 0, 1],
                       [1, 0, 0, 0, 0, 1, 0]])
    W = preprocessing.normalize(W, norm="l1", axis=1)

    H = sp.csr_matrix(np.random.random((W.shape[0], 3)))
    H = preprocessing.normalize(H, norm="l1", axis=0)
    L = sp.dia_matrix((H.sum(axis=0), [0]), shape=(H.shape[1], H.shape[1]))
    for i in range(40):
        HLHT = H * L * H.T
        print("Iteration %s: %s" % (i, divergence(W, HLHT)))
        H, L = update(W, H, L)
    print(HLHT.todense())
    print(H.todense())


if __name__ == "__main__":
    cluster_figure1a()
