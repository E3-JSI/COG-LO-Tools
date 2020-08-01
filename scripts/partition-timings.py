import sys
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

import modules.partitioning.spectral as part
import modules.partitioning.utils as part_utils


def read_dist_matrix(dirname, graph_size, trialN):
    print('reading distance matrix of size ' + str(graph_size) + ', trial: ' + str(trialN))
    print('reading node file')
    node_fpath = os.path.join(dirname, 'nodes-' + str(graph_size) + '-' + str(trialN) + '.csv')
    assign_vec = []
    with open(node_fpath, 'r') as f_in:
        for line in f_in.readlines():
            spl = line.split(',')
            assign_vec.append(int(spl[2]))
    assign_vec = np.array(assign_vec, dtype=int)

    n_nodes = len(assign_vec)

    print('reading matrix file')
    mat_fpath = os.path.join(dirname, 'dist-' + str(graph_size) + '-' + str(trialN) + '.csv')
    line_vec = []
    with open(mat_fpath, 'r') as f_in:
        for line in f_in.readlines():
            spl = line.split(',')
            line_vec.append(spl)

    print('constructing the graph matrix')
    n_entries = len(line_vec)

    val_vec = np.empty(n_entries)
    row_idxs = np.empty(n_entries, dtype=int)
    col_idxs = np.empty(n_entries, dtype=int)
    for valN, spl in enumerate(line_vec):
        val_vec[valN] = float(spl[2])
        row_idx = float(spl[0])
        col_idx = float(spl[1])

        if row_idx % 1 != 0:
            raise ValueError('Invalid index: ' + str(row_idx))
        if col_idx % 1 != 0:
            raise ValueError('Invalid index: ' + str(col_idx))

        row_idxs[valN] = int(row_idx) - 1
        col_idxs[valN] = int(col_idx) - 1

    D = sparse.csr_matrix((val_vec, (row_idxs, col_idxs)), shape=(n_nodes, n_nodes))

    return D

def read_time_matrix(dirname, graph_size, trialN):
    print('reading time matrix of size ' + str(graph_size) + ', trial: ' + str(trialN))
    print('reading node file')
    node_fpath = os.path.join(dirname, 'nodes-' + str(graph_size) + '-' + str(trialN) + '.csv')
    assign_vec = []
    with open(node_fpath, 'r') as f_in:
        for line in f_in.readlines():
            spl = line.split(',')
            assign_vec.append(int(spl[2]))
    assign_vec = np.array(assign_vec, dtype=int)

    n_nodes = len(assign_vec)

    print('reading matrix file')
    mat_fpath = os.path.join(dirname, 'time-' + str(graph_size) + '-' + str(trialN) + '.csv')
    line_vec = []
    with open(mat_fpath, 'r') as f_in:
        for line in f_in.readlines():
            spl = line.split(',')
            line_vec.append(spl)

    print('constructing the graph matrix')
    n_entries = len(line_vec)

    val_vec = np.empty(n_entries)
    row_idxs = np.empty(n_entries, dtype=int)
    col_idxs = np.empty(n_entries, dtype=int)
    for valN, spl in enumerate(line_vec):
        val_vec[valN] = float(spl[2])
        row_idx = float(spl[0])
        col_idx = float(spl[1])

        if row_idx % 1 != 0:
            raise ValueError('Invalid index: ' + str(row_idx))
        if col_idx % 1 != 0:
            raise ValueError('Invalid index: ' + str(col_idx))

        row_idxs[valN] = int(row_idx) - 1
        col_idxs[valN] = int(col_idx) - 1

    T = sparse.csr_matrix((val_vec, (row_idxs, col_idxs)), shape=(n_nodes, n_nodes))

    return T



def read_matrix(dirname, graph_size, trialN):
    print('reading graph of size ' + str(graph_size) + ', trial: ' + str(trialN))
    print('reading node file')
    node_fpath = os.path.join(dirname, 'nodes-' + str(graph_size) + '-' + str(trialN) + '.csv')
    assign_vec = []
    with open(node_fpath, 'r') as f_in:
        for line in f_in.readlines():
            spl = line.split(',')
            assign_vec.append(int(spl[2]))
    assign_vec = np.array(assign_vec, dtype=int)

    n_nodes = len(assign_vec)

    print('reading matrix file')
    mat_fpath = os.path.join(dirname, 'time-' + str(graph_size) + '-' + str(trialN) + '.csv')
    line_vec = []
    with open(mat_fpath, 'r') as f_in:
        for line in f_in.readlines():
            spl = line.split(',')
            line_vec.append(spl)

    print('constructing the graph matrix')
    n_entries = len(line_vec)

    val_vec = np.empty(n_entries)
    row_idxs = np.empty(n_entries, dtype=int)
    col_idxs = np.empty(n_entries, dtype=int)
    for valN, spl in enumerate(line_vec):
        val_vec[valN] = float(spl[2])
        row_idx = float(spl[0])
        col_idx = float(spl[1])

        if row_idx % 1 != 0:
            raise ValueError('Invalid index: ' + str(row_idx))
        if col_idx % 1 != 0:
            raise ValueError('Invalid index: ' + str(col_idx))

        row_idxs[valN] = int(row_idx) - 1
        col_idxs[valN] = int(col_idx) - 1

    Q = sparse.csr_matrix((val_vec, (row_idxs, col_idxs)), shape=(n_nodes, n_nodes))

    print('reading the cluster assignment')

    print('inverting travel times')
    # invert the times to get intensities
    np.reciprocal(Q.data, out=Q.data)

    print('matrix constructed')

    return Q, assign_vec


def compute_pairwise_dist(D_part):
    n_rows = D_part.shape[0]
    n_cols = n_rows
    for rowN in range(n_rows):
        for colN in range(n_cols):
            if rowN == colN:
                continue
            if D_part[rowN, colN] == 0:
                D_part[rowN, colN] = np.inf

    D_full = part_utils.compute_pairwise_dist(D_part)
    return D_full


def write_pairwise_mat(D, dirname, prefix, graph_size, trialN, clustN):
    fname = prefix + '-' + str(graph_size) + '-' + str(trialN) + '-part-' + str(clustN) + '.csv'
    fpath = os.path.join(dirname, fname)

    print('writing file: ' + fname)

    n_nodes = D.shape[0]
    with open(fpath, 'w') as f_out:
        for rowN in range(n_nodes):
            for colN in range(n_nodes):
                if rowN == colN:
                    continue
                row_str = str(rowN) + ',' + str(colN) + ',' + str(D[rowN, colN])
                f_out.write(row_str + '\n')



def dump_pairwise_dist_mat(part_vec, n_clusts, dirname, graph_size, trialN):
    # read the distance matrix
    D = read_dist_matrix(dirname, graph_size, trialN)
    T = read_time_matrix(dirname, graph_size, trialN)

    np.set_printoptions(threshold=sys.maxsize)

    for clustN in range(n_clusts):
        part_idxs = part_vec == clustN

        # transform the distance matrix
        D_part_sparse = D[part_idxs, :][:, part_idxs]
        D_part = D_part_sparse.todense()
        D_full = compute_pairwise_dist(D_part)
        write_pairwise_mat(D_full, dirname, 'dist', graph_size, trialN, clustN)
        # transform the time matrix
        T_part_sparse = T[part_idxs, :][:, part_idxs]
        T_part = T_part_sparse.todense()
        T_full = compute_pairwise_dist(T_part)
        write_pairwise_mat(D_full, dirname, 'time', graph_size, trialN, clustN)


def main():
    dirname = '~/data/coglo/sim-graphs/'
    if dirname[0] == '~':
        home_dir = str(Path.home())
        dirname = os.path.join(home_dir, dirname[2:])

    size_vec = [
        1000,
        2000,
        5000,
        7000,
        10000,
        20000,
        50000
    ]
    n_trials = 10

    for graph_size in size_vec:
        for trialN in range(1, n_trials+1):
            Q, clust_vec = read_matrix(dirname, graph_size, trialN)

            n_clusts = len(np.unique(clust_vec))

            print('clustering')
            start_sec = time.time()
            part_vec, _ = part.spectral_part_sparse(Q, n_clusts, sparse_la=True, debug_assert=False)
            dur_sec = time.time() - start_sec
            print('partitioning finished in ' + str(round(dur_sec, 1)) + ' seconds')

            cut_size_orig = part_utils.cut_size(Q, clust_vec, n_clusts)
            cut_size = part_utils.cut_size(Q, part_vec, n_clusts)
            # print('partition: ' + str([str(val) for val in part_vec]))
            print('cut size: ' + str(cut_size) + ', original cut size: ' + str(cut_size_orig))

            dump_pairwise_dist_mat(part_vec, n_clusts, dirname, graph_size, trialN)

            part_fname = os.path.join(dirname, 'part-' + str(graph_size) + '-' + str(trialN) + '.csv')
            with open(part_fname, 'w') as f_out:
                f_out.write('\n'.join([str(val) for val in part_vec]))

            timings_fname = os.path.join(dirname, 'performance.csv')
            with open(timings_fname, 'a') as f_out:
                f_out.write(str(graph_size) + '-' + str(trialN) + ',' + str(cut_size) + ',' + str(cut_size_orig) + ',' + str(dur_sec) + '\n')



if __name__ == '__main__':
    main()
