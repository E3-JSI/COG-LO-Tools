import scipy

from scipy.sparse import csr_matrix
import numpy as np

import modules.partitioning.dijkstra as dijkstra

def laplace_mat(A_spmat):
    '''
    Returns the Laplace matrix of the given adjacency matrix A.
    '''
    nvert = A_spmat.get_shape()[0]

    range_n = scipy.array(range(nvert))
    ones_n = scipy.array([1 for _ in range_n])

    # d <- A*1
    diag_vec = csr_matrix.dot(A_spmat, ones_n)
    # D <- diag(d)
    deg_spmat = csr_matrix((diag_vec, (range_n, range_n)), shape=(nvert, nvert))

    # L <- D - A
    laplace_spmat = (deg_spmat - A_spmat)
    # convert L doubles (if the matrix is already
    # a floating point type, the call only returns a reference
    # to self)
    laplace_spmat = laplace_spmat.asfptype()

    return laplace_spmat


def cut_size(Q, part_vec, k):
    n = Q.shape[0]

    P = np.empty((n, k))
    ones_k = np.ones(k)

    # compute the cut intensity
    for partN in range(k):
        P[:, partN] = part_vec == partN

    Pt_Q = Q.__rmul__(P.T)
    Q_p = np.dot(Pt_Q, P)
    # cut_sum = cut_intensity_sparse(Q, part, k)
    # k is small, so optimization here is not as critical
    cut_intensity = np.dot(np.dot(ones_k, Q_p), ones_k) - np.trace(Q_p)

    return cut_intensity

def cut_size_undirected(A_spmat, partitions):
    return 0.5*cut_size(A_spmat, partitions)


def compute_pairwise_dist(D):
    n_nodes = D.shape[0]

    D_dense = np.empty((n_nodes, n_nodes))

    for start_nodeN in range(n_nodes):
        node_dist_vec = dijkstra.run_dijkstra(D, start_nodeN)
        D_dense[start_nodeN, :] = node_dist_vec

    return D_dense

