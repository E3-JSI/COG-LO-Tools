import unittest

from scipy.sparse import csr_matrix

from modules.partitioning.recursive_bipart import RecursiveBipart
from modules.partitioning.utils import cut_size_undirected

import modules.utils.ConflictNodeOrdering as node_ordering

class TestConflictOrdering(unittest.TestCase):

    def test_basic(self):
        relations = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (4, 2), (5, 3)]

        expected = [1, 2, 4, 2, 5, 3]

        result = node_ordering.OrderRelations.order_relations(relations)

        for src_id, dst_id in relations:
            try:
                # check that there is an occurence of dst_id after src_id
                srcN = result.index(src_id)
                dstN = result.index(dst_id, srcN+1)

                self.assertTrue(srcN < dstN)
            except Exception as e:
                # if an ID is not found an exception is raised
                self.assertTrue(False)



if __name__ == "__main__":
    unittest.main()
