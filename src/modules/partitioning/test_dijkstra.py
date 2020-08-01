import unittest

import numpy as np

import modules.partitioning.dijkstra as dijkstra


class TestDijkstra(unittest.TestCase):

    def test_dijsktra_small(self):

        D = np.array([
            [0,  1,   3,      np.inf, np.inf],
            [1,  0,   np.inf,    4,   2],
            [3,  np.inf,     0,   1,      np.inf],
            [np.inf,    4,  1, 0,   np.inf],
            [np.inf,    2,  np.inf, np.inf, 0]
        ])
        dist_vec = dijkstra.run_dijkstra(D, 0)

        real_dist_vec = np.array([
            0,
            1,
            3,
            4,
            3
        ])

        for distN, dist in enumerate(real_dist_vec):
            self.assertEqual(dist, dist_vec[distN])

    def test_dijkstra_medium(self):

        i = np.inf

        D = np.array([
            [0, 2, i, i, i, i, i, i, 1, 1, i, i, 9],
            [2, 0, 1, i, i, i, i, i, i, i, i, i, i],
            [i, 1, 0, 1, i, i, i, i, 4, i, i, i, i],
            [i, i, 1, 0, 1, i, i, i, i, i, i, i, i],
            [i, i, i, 1, 0, 2, 1, i, i, i, i, i, i],
            [i, i, i, i, 2, 0, i, i, 1, i, i, i, i],
            [i, i, i, i, 1, i, 0, 1, i, i, i, i, i],
            [i, i, i, i, i, i, 1, 0, 1, i, i, i, i],
            [1, i, i, i, i, 1, i, 1, 0, i, i, i, i],
            [1, i, i, i, i, i, i, i, i, 0, 1, i, i],
            [i, i, i, i, i, i, i, i, i, 1, 0, 1, i],
            [i, i, i, i, i, i, i, i, i, i, 1, 0, 1],
            [9, i, i, i, i, i, i, i, i, i, i, 1, 0],
        ])
        dist_vec = dijkstra.run_dijkstra(D, 0)

        real_dist_vec = np.array([
            0,
            2,
            3,
            4,
            4,
            2,
            3,
            2,
            1,
            1,
            2,
            3,
            4
        ])

        for distN, dist in enumerate(real_dist_vec):
            self.assertEqual(dist, dist_vec[distN])


if __name__ == '__main__':
    unittest.main()

