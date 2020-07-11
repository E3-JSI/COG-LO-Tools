import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

import modules.partitioning.spectral as part
import modules.partitioning.utils as part_utils


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




def main():
    dirname = '~/data/coglo/sim-graphs/'
    if dirname[0] == '~':
        home_dir = str(Path.home())
        dirname = os.path.join(home_dir, dirname[2:])

    size_vec = [
        1000,
        5000,
        10000,
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

            part_fname = os.path.join(dirname, 'part-' + str(graph_size) + '-' + str(trialN) + '.csv')
            with open(part_fname, 'w') as f_out:
                f_out.write('\n'.join([str(val) for val in part_vec]))

            timings_fname = os.path.join(dirname, 'performance.csv')
            with open(timings_fname, 'a') as f_out:
                f_out.write(str(graph_size) + '-' + str(trialN) + ',' + str(cut_size) + ',' + str(cut_size_orig) + ',' + str(dur_sec) + '\n')



if __name__ == '__main__':
    main()
