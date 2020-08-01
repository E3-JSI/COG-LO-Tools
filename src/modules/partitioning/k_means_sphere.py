import numpy as np

import scipy.sparse as sparse
import scipy.sparse.linalg as sp_linalg


def discretesample(p, n):
    """independently draws n samples (with replacement) from the
        distribution specified by p, where p is a probability array
        whose elements sum to 1."""
    return np.random.choice(len(p), n, p=p)


def similarity(x1, x2):
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    return np.dot(x1, x2) / (norm_x1 * norm_x2)


def calc_centroid(X):
    return np.mean(X, 0)


def count_diff(a, prev_a):
    diff = np.abs(a - prev_a)
    diff[diff > 1] = 1

    return np.sum(diff)


def k_means_sphere(X, k, max_iter=100):
    # kmeans++ init
    n = X.shape[0]

    def kmeanspp(_X, _k):
        n_inst = _X.shape[0]
        dim = _X.shape[1]

        # select initial centroids
        C = np.zeros((_k, dim))  # k++ means
        rands = np.random.randint(0, n_inst)
        C[0, :] = _X[rands]

        probs = np.zeros(n_inst)

        for centroidN in range(1, _k):
            # compute probabilities for new ctroid
            for recN in range(0, n_inst):
                rec = _X[recN, :]

                # compute distance to nearest centroid
                nearest_dist = np.inf
                for exist_centroidN in range(0, centroidN):
                    centroid = C[exist_centroidN, :]
                    clust_rec_sim = similarity(rec, centroid)
                    clust_rec_dist = 0.5 * (1 - clust_rec_sim)

                    if clust_rec_dist < nearest_dist:
                        nearest_dist = clust_rec_dist
                # the probability is proportional to d^2
                probs[recN] = nearest_dist * nearest_dist
            norm_factor = 1.0 / np.sum(probs)
            probs = probs * norm_factor

            chosenN = discretesample(probs, 1)[0]
            C[centroidN, :] = _X[chosenN, :]
            print('Chosen centroid {}, probability: {}, max probability: {} '.format(chosenN, probs[centroidN],
                                                                                     probs.max()))
        return C

    C = kmeanspp(X, k)
    prev_assignment = np.zeros(n)
    assignment = np.zeros(n)

    change = True
    iterN = 0

    while change and iterN < max_iter:
        iterN += 1
        lost_centroid = True

        while lost_centroid:
            lost_centroid = False

            # assign vectors
            for recN in range(0, n):
                xi = X[recN, :]
                best_idx = -1
                best_sim = np.NINF

                for clustN in range(0, k):
                    sim = similarity(xi, C[clustN, :])
                    if sim > best_sim:
                        best_idx = clustN
                        best_sim = sim

                assignment[recN] = best_idx

            # recompute centroids
            for clustN in range(0, k):
                assigned_idxs = assignment == clustN

                if assigned_idxs.astype(dtype=int).sum() > 0:
                    Yn = X[assigned_idxs, :]
                    C[clustN, :] = calc_centroid(Yn)
                else:
                    C = kmeanspp(X, k)
                    lost_centroid = True
                    print("Lost a centroid, reinitialized at {}".format(iterN))
                    break

        diff = count_diff(assignment, prev_assignment)
        change = diff > 0
        tmp = prev_assignment
        prev_assignment = assignment
        assignment = tmp

    return assignment


def k_means_sphere_vectorized(X, k, max_iter=100, debug_assert=False):
    print('running k-means')

    # dimensions and other variables
    n = X.shape[0]
    d = X.shape[1]

    range_n = range(n)
    range_k = range(k)

    # kmeans++ init
    def kmeanspp(_X, _X_norm, _k):
        n_inst = _X_norm.shape[0]
        dim = _X_norm.shape[1]

        # select initial centroids
        C = np.empty((_k, dim))  # k++ means
        C_norm_t = np.empty((dim, _k))

        rands = np.random.randint(0, n_inst)
        C[0, :] = _X[rands, :]
        C_norm_t[:, 0] = _X_norm[rands, :]

        select_prob_vec = np.empty(n_inst)
        mn_clust_dist_vec = np.empty(n_inst)

        for centroidN in range(1, _k):
            # compute probabilities for the new centroid
            C_curr_norm_t = C_norm_t[:, :centroidN]

            # compute the distance as 0.5*(1 - sim)
            ftrv_clust_dist_mat = np.dot(_X_norm, C_curr_norm_t)
            np.subtract(1, ftrv_clust_dist_mat, out=ftrv_clust_dist_mat)
            np.multiply(0.5, ftrv_clust_dist_mat, out=ftrv_clust_dist_mat)

            # for each of the examples compute the distance to the nearest centroid
            np.min(ftrv_clust_dist_mat, axis=1, out=mn_clust_dist_vec)

            # the probability of selecting the instance should be
            # proportional to d^2
            np.multiply(mn_clust_dist_vec, mn_clust_dist_vec, select_prob_vec)

            prob_sum = np.sum(select_prob_vec)
            if prob_sum < 1e-12:
                np.multiply(np.ones(n_inst), 1 / n_inst, out=select_prob_vec)
                prob_sum = n_inst

            norm_factor = 1.0 / prob_sum
            np.multiply(select_prob_vec, norm_factor, out=select_prob_vec)

            # sample the new centroid according to the computed distribution
            chosenN = discretesample(select_prob_vec, 1)[0]
            C[centroidN, :] = _X[chosenN, :]
            C_norm_t[:, centroidN] = _X_norm[chosenN, :]

        return C

    # normalize the rows of X so it will be faster to compute the cosine
    # similarity
    one_by_row_norm_X_vec = np.linalg.norm(X, axis=1)
    np.reciprocal(one_by_row_norm_X_vec, out=one_by_row_norm_X_vec)
    Dx = sparse.csr_matrix((one_by_row_norm_X_vec, (range_n, range_n)), shape=(n, n))
    X_norm = Dx.dot(X)
    del one_by_row_norm_X_vec
    del Dx

    if debug_assert:
        # check that the rows of X are of unit length
        row_norm_vec = np.linalg.norm(X_norm, axis=1)
        assert np.linalg.norm(np.ones(n) - row_norm_vec) < 1e-7
        assert isinstance(X_norm, np.ndarray)

    C = kmeanspp(X, X_norm, k)
    prev_assignment = -np.ones(n, dtype=int)
    assignment_vec = np.empty(n, dtype=int)
    assign_diff_vec = np.empty(n, dtype=int)

    clust_sim_ftrvv = np.empty((n, k))
    clust_sum_vec = np.empty(d)

    change = True
    iterN = 0

    while change and iterN < max_iter:
        iterN += 1
        lost_centroid = True

        if iterN % 10 == 0:
            print('iterN: ' + str(iterN))

        # normalize the centroids to make the computation of the cosine
        # similarity faster
        inv_row_norm_C_vec = np.linalg.norm(C, axis=1)
        np.reciprocal(inv_row_norm_C_vec, out=inv_row_norm_C_vec)
        D_one_by_norm = sparse.csr_matrix((inv_row_norm_C_vec, (range_k, range_k)), shape=(k, k))
        C_norm_t = D_one_by_norm.dot(C).T

        del inv_row_norm_C_vec
        del D_one_by_norm

        if debug_assert:
            # check that the columns of C are of unit length
            col_norm_vec = np.linalg.norm(C_norm_t, axis=0)
            assert np.linalg.norm(np.ones(k) - col_norm_vec) < 1e-7
            assert isinstance(C_norm_t, np.ndarray)

        while lost_centroid:
            lost_centroid = False

            # assign vectors
            np.dot(X_norm, C_norm_t, out=clust_sim_ftrvv)
            # the assignment is the index of the maximal value
            # in each row
            np.argmax(clust_sim_ftrvv, axis=1, out=assignment_vec)

            # recompute centroids
            for clustN in range(0, k):
                assigned_idxs = assignment_vec == clustN

                clust_size = np.sum(assigned_idxs)

                if clust_size > 0:
                    np.dot(assigned_idxs, X, out=clust_sum_vec)
                    np.multiply(1 / clust_size, clust_sum_vec, out=C[clustN, :])
                    # centroid_ftrvv = (1 / clust_size) * clust_sum_vec
                    # C[clustN, :] = centroid_vec
                else:
                    C = kmeanspp(X_norm, k)
                    lost_centroid = True
                    print("Lost a centroid, reinitialized at {}".format(iterN))
                    break

        # check if the assignments changed
        np.subtract(assignment_vec, prev_assignment, out=assign_diff_vec)
        np.abs(assign_diff_vec, out=assign_diff_vec)
        change = np.sum(assign_diff_vec) > 0

        # swap the assignment and prev_assignment vectors
        tmp = prev_assignment
        prev_assignment = assignment_vec
        assignment_vec = tmp

    return assignment_vec
