import unittest

import numpy as np
import scipy.sparse as sparse

import modules.partitioning.spectral as clust


class TestSparse(unittest.TestCase):

    def test_sparse_basic(self):
        np.random.seed(1)

        Q = sparse.csr_matrix(np.array([
            [0, 2, 3, 0, 0, 0],
            [3, 0, 1, 0, 0, 0],
            [2, 1, 0, 0.1, 0, 0],
            [0, 0, 0.1, 0, 3, 2],
            [0, 0, 0, 1, 0, 3],
            [0, 0, 0, 2, 1, 0],
        ]))
        k = 2

        part_vec, Y = clust.spectral_part_sparse(Q, k, debug_assert=True)

        self.assertTrue(part_vec[0] == part_vec[1])
        self.assertTrue(part_vec[0] == part_vec[2])
        self.assertTrue(part_vec[0] != part_vec[3])
        self.assertTrue(part_vec[0] != part_vec[4])
        self.assertTrue(part_vec[0] != part_vec[5])

    def test_sparse_3_clusts(self):
        np.random.seed(1)

        Q = sparse.csr_matrix(np.array([
            [0, 2, 3, 0, 0, 0, 0, 0, 0],
            [3, 0, 1, 0, 0, 0, 0, 0, 0],
            [2, 1, 0, 0.1, 0, 0, 0, 0, 0],
            [0, 0, 0.1, 0, 3, 2, 0, 0, 0],
            [0, 0, 0, 1, 0, 3, 0, 0, 0],
            [0, 0, 0, 2, 1, 0, 0.05, 0, 0],
            [0, 0, 0, 0, 0, 0.05, 0, 5, 5],
            [0, 0, 0, 0, 0, 0, 8, 0, 3],
            [0, 0, 0, 0, 0, 0, 4, 9, 0],
        ]))
        k = 3

        part_vec, Y = clust.spectral_part_sparse(Q, k, debug_assert=True)

        # cluster 1
        self.assertTrue(part_vec[0] == part_vec[1])
        self.assertTrue(part_vec[0] == part_vec[2])
        self.assertTrue(part_vec[0] != part_vec[3])
        self.assertTrue(part_vec[0] != part_vec[4])
        self.assertTrue(part_vec[0] != part_vec[5])
        self.assertTrue(part_vec[0] != part_vec[6])
        self.assertTrue(part_vec[0] != part_vec[7])
        self.assertTrue(part_vec[0] != part_vec[8])
        # cluster 2
        self.assertTrue(part_vec[3] == part_vec[4])
        self.assertTrue(part_vec[3] == part_vec[5])
        self.assertTrue(part_vec[3] != part_vec[6])
        self.assertTrue(part_vec[3] != part_vec[7])
        self.assertTrue(part_vec[3] != part_vec[8])


if __name__ == '__main__':
    unittest.main()

