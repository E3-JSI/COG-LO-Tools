import numpy as np
from modules.partitioning.k_means_sphere import k_means_sphere, k_means_sphere_vectorized

import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_linalg


def cut_intensity(Q, part, n_part):
    n_vert = Q.shape[0]
    ones_k = np.ones((n_part, 1))
    P = np.zeros((n_vert, n_part))
    for partN in range(n_part):
        P[:, partN] = part == partN

    Q_p = np.dot(np.dot(P.T, Q), P)
    c = np.dot(np.dot(ones_k.T, Q_p), ones_k) - np.trace(Q_p)
    return c


def cut_intensity_sparse(Q, part, n_part):
    n_vert = Q.shape[0]

    ones_k = np.ones(n_part)
    P = np.zeros((n_vert, n_part))

    for partN in range(n_part):
        P[:, partN] = part == partN

    Pt_Q = Q.__rmul__(P.T)
    Q_p = np.dot(Pt_Q, P)
    c = np.dot(np.dot(ones_k, Q_p), ones_k) - np.trace(Q_p)
    return c


def spectral_part(Q, k):
    """
        % A spectral partitoning algorithm that
    % splits a graph into k partitions by
    % approximating the minimal cut.
    %
    % The graph is represented as an intensity
    % matrix Q. An edge (Q)_ij is the rate of
    % going from node i to node j.
    %
    % The method returns a vector of assignments
    % a where a_i is the index of the parition
    % to which the i-th element was assigned.

    """
    n = Q.shape[0]

    # symmetrize to ensure real eigen vals
    Q = 0.5 * (Q + Q.T)
    Q_norm = Q / np.sum(Q, 1, keepdims=True)  # for broadcast fix

    # compute Laplace matrix
    L = np.eye(n) - Q_norm

    # calc eigenvectors
    [v, V] = np.linalg.eig(L)

    # sorting eigenvals and vecs
    I = np.argsort(v)
    V = V[:, I]
    del I

    # create datase Y, each row is a feature vector for associated node in graph
    X = V[:, 0:k]
    Y = X[:, 1:]

    norms = np.sqrt(np.sum(Y * Y, 1, keepdims=True))
    Y_norm = Y / norms

    best_part = 0
    best_cut = np.inf

    total_tries = 10  # attempt multiple tries to get best clustering - due to randomness
    for i in range(total_tries):
        part = k_means_sphere(Y_norm, k, 200)

        # clustering evaluation
        cut_sum = cut_intensity(Q, part, k)

        if cut_sum < best_cut:
            print('Got better clustering in try', i)
            best_cut = cut_sum
            best_part = part

    return best_part, Y


def spectral_part_sparse(Q, k, sparse_la=True, debug_assert=False):
    """
    % A spectral partitoning algorithm that
    % splits a graph into k partitions by
    % approximating the minimal cut.
    %
    % The graph is represented as an intensity
    % matrix Q. An edge (Q)_ij is the rate of
    % going from node i to node j.
    %
    % The method returns a vector of assignments
    % a where a_i is the index of the parition
    % to which the i-th element was assigned.

    """
    if not isinstance(Q, sparse.csr_matrix):
        raise ValueError('The input matrix Q should be a scipy.sparse.csr_matrix!')
    if Q.shape[0] != Q.shape[1]:
        raise ValueError('The input matrix Q should be square!')

    n = Q.shape[0]

    # re-used variables
    ones_n = np.ones(n)
    ones_k = np.ones(k)
    range_n = range(n)

    # symmetrize to ensure real eigen vals
    Q = (Q + Q.getH()) * 0.5

    if debug_assert:
        diag_vec = Q.diagonal()
        assert np.linalg.norm(diag_vec) < 1e-7

    # compute Laplace matrix
    D_row_sum = sparse.csr_matrix((Q.dot(ones_n), (range_n, range_n)), shape=(n, n))
    L = D_row_sum - Q
    del D_row_sum

    if debug_assert:
        # check that the rows of L sum to 0
        row_sum_vec = L.dot(ones_n)
        assert np.linalg.norm(row_sum_vec) < 1e-7
        assert isinstance(L, sparse.csr_matrix)

    # calc eigenvalues and eigenvectors
    print('computing ' + str(k-1) + ' eigenvectors')
    if sparse_la:
        eigval_vec, V = sparse.linalg.eigsh(
            L,
            k=k,
            which='SA',
            return_eigenvectors=True
        )

        if debug_assert:
            assert len(eigval_vec) == k
            assert abs(eigval_vec[0]) < 1e-9, 'The first eigenvalue is not zero: ' + str(eigval_vec[0]) + '\n' + str(eigval_vec)
            for valN in range(1, k):
                eigval_vec[valN-1] <= eigval_vec[valN]

        print('preparing the new dataset')
        Y = V[:, 1:]
    else:
        eigval_vec, V = scipy.linalg.eigh(
            L.todense(),
            overwrite_a=True,
            eigvals=(1, k-1)
            # subset_by_index=(1, k-1)
        )

        if debug_assert:
            assert len(eigval_vec) == k-1
            for valN in range(k-1):
                eigval_vec[valN-1] <= eigval_vec[valN]

        Y = V

    # create dataset Y where each row is a feature vector for associated node in graph

    if debug_assert:
        assert Y.shape[1] == k - 1
        # check that Y is a dense array
        assert isinstance(Y, np.ndarray)

    # normalize the rows of Y to make them unit length
    inv_row_norm_Y_vec = np.linalg.norm(Y, axis=1)
    np.reciprocal(inv_row_norm_Y_vec, out=inv_row_norm_Y_vec)
    D_one_by_norm = sparse.csr_matrix((inv_row_norm_Y_vec, (range_n, range_n)), shape=(n, n))
    Y_norm = D_one_by_norm.dot(Y)

    del inv_row_norm_Y_vec
    del D_one_by_norm

    if debug_assert:
        # check that the rows of X are of unit length
        row_norm_vec = np.linalg.norm(Y_norm, axis=1)
        assert np.linalg.norm(ones_n - row_norm_vec) < 1e-7

    # re-used variables
    P = np.empty((n, k))
    Q_p = np.empty((k, k))

    best_part = 0
    best_cut = np.inf

    total_tries = 10  # attempt multiple tries to get best clustering - due to randomness
    for trialN in range(total_tries):
        part = k_means_sphere_vectorized(Y_norm, k, 200, debug_assert=debug_assert)

        # compute the cut intensity
        for partN in range(k):
            P[:, partN] = part == partN

        Pt_Q = Q.__rmul__(P.T)
        np.dot(Pt_Q, P, out=Q_p)
        # cut_sum = cut_intensity_sparse(Q, part, k)
        # k is small, so optimization here is not as critical
        cut_intensity = np.dot(np.dot(ones_k, Q_p), ones_k) - np.trace(Q_p)

        if cut_intensity < best_cut:
            print('Got better clustering in try', trialN)
            best_cut = cut_intensity
            best_part = part

    return best_part, Y
