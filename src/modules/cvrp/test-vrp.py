#!/bin/bash

from vrp import VRP
import unittest
vrp = VRP

def getEdgeNameVec(graph_incidence_mat):
    edge_name_vec = []
    for colN in range(len(graph_incidence_mat[0])):
        edge_str = ''
        for rowN, row in enumerate(graph_incidence_mat):
            if row[colN] != 0:
                edge_str += str(rowN+1)
        edge_name_vec.append(edge_str)
    return edge_name_vec

class TestSum(unittest.TestCase):

    def testFourNodesTwoVehicles(self):
        print('=======================================')
        print('TEST 4 NODES 2 VEHICLES')
        print('=======================================')
        # read file test-vrp
        ## V1: location 0, 1(2)
        ## V2: location: 0, 3(2)
        dispatch_vec = [0, 3, 4, 4]
        capacity_vec = [8, 4]
        start_loc_vec = [[0], [0]]
        capacity_vec[0] -= 2
        capacity_vec[1] -= 2
        start_loc_vec[0].append(1)
        start_loc_vec[1].append(3)

        # 4---3
        # |   |
        # 1---2
        # network graph, columns = edges 12, 23, 13, vrstice = mesta
        graph = [
                [1, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1]
        ]
        edges_length = [1,1,1,1]

        route_mat, dispatch_mat, obj_val = vrp.vrp(graph, dispatch_vec, capacity_vec, start_loc_vec, edges_length)

        route_name_vec = getEdgeNameVec(graph)

        print('obj val: ' + str(obj_val))
        print("routes:")
        for row in route_mat:
            print(str([route_name_vec[edgeN] + ': ' + str(int(row[edgeN])) for edgeN in range(len(row))]))
        print("dispatch:")
        for row in dispatch_mat:
            print(str(row))

        #assert obj_val == 4
        #assert sum(route_mat[0]) == 2
        #assert sum(route_mat[1]) == 2
        #assert sum(dispatch_mat[0]) == 4
        #assert sum(dispatch_mat[1]) == 3
        print('=======================================')
        print("route1:", route_mat[0])
        print("route2:", route_mat[1])
        print("dispatch1:", dispatch_mat[0])
        print("dispatch2:", dispatch_mat[1])


    """
    def test4Nodes1Vehicle(self):
        print('=======================================')
        print('TEST 4 NODES 1 VEHICLE')
        print('=======================================')
        # read file test-vrp
        dispatch_vec = [1, 3, 1, 4]
        capacity_vec = [9]
        start_loc_vec = [0]

        # 4---3
        # |   |
        # 1---2
        graph = [
                [1, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1]
        ]

        route_mat, dispatch_mat, obj_val = vrp(graph, dispatch_vec, capacity_vec, start_loc_vec)

        route_name_vec = getEdgeNameVec(graph)

        print('obj val: ' + str(obj_val))
        print("routes:")
        for row in route_mat:
            print(str([route_name_vec[edgeN] + ': ' + str(int(row[edgeN])) for edgeN in range(len(row))]))

        print("node dispatch:")
        for row in dispatch_mat:
            print(str(row))

        assert obj_val == 4
        assert sum(route_mat[0]) == 4
        assert route_mat[0][0] + route_mat[0][1] > 0
        assert sum(dispatch_mat[0]) == 9

        print('=======================================')
        print()
        print()

    def test3NodesOneVehicle(self):
        print('=======================================')
        print('TEST 3 NODES 1 VEHICLE')
        print('=======================================')
        # read file test-vrp
        dispatch_vec = [1, 3, 1]
        capacity_vec = [5]
        start_loc_vec = [0]

        #     3
        #   / |
        # 1---2
        # network graph, columns = edges 12, 23, 13, vrstice = mesta
        graph = [
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 1]
        ]

        route_mat, dispatch_mat, obj_val = vrp(graph, dispatch_vec, capacity_vec, start_loc_vec)

        route_name_vec = getEdgeNameVec(graph)

        print('obj val: ' + str(obj_val))
        print("routes:")
        for row in route_mat:
            print(str([route_name_vec[edgeN] + ': ' + str(int(row[edgeN])) for edgeN in range(len(row))]))
        print("dispatch:")
        for row in dispatch_mat:
            print(str(row))

        assert obj_val == 3
        assert sum(route_mat[0]) == 3
        assert sum(dispatch_mat[0]) == 5
        print('=======================================')
        print()
        print()

    def test6Nodes1Vehicle(self):
        print('=======================================')
        print('TEST 6 NODES 1 VEHICLE')
        print('=======================================')
        # read file test-vrp
        dispatch_vec = [11, 22, 33, 44, 55, 66]
        capacity_vec = [300]
        start_loc_vec = [0]


        # TEST:
        # 6---5---4
        # | / | / |
        # 1---2---3
        # VEHICLES: 1, capacity: 300
        # total load: 231

        # network graph, stopci = povezave 12, 23, 13, vrstice = mesta
        graph = [
            [1, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 0, 0, 0]
        ]

        route_mat, dispatch_mat, obj_val = vrp(graph, dispatch_vec, capacity_vec, start_loc_vec)

        route_name_vec = getEdgeNameVec(graph)

        print('obj val: ' + str(obj_val))
        print("routes:")
        for row in route_mat:
            print(str([route_name_vec[edgeN] + ': ' + str(int(row[edgeN])) for edgeN in range(len(row))]))

        print("dispatch:")
        for row in dispatch_mat:
            print(str(row))

        assert len(dispatch_mat) == 1
        for valN, val in enumerate(dispatch_mat[0]):
            assert abs(val - dispatch_vec[valN]) < 1e-7
        assert len(route_mat) == 1
        assert sum(route_mat[0]) == 6

        print("test #2 finished")
        print('=======================================')
        print()
        print()


    def test6Nodes3Vehicles(self):
        print('=======================================')
        print('TEST 6 NODES 3 VEHICLES')
        print('=======================================')
        # TEST:
        # 6---5---4
        # | / | / |
        # 1---2---3
        # VEHICLES: 1, capacity: 240
        # total load: 231
        # read file test-vrp

        dispatch_vec = [11, 22, 33, 44, 55, 66]
        capacity_vec = [80, 80, 80]
        start_loc_vec = [0,0,0]

        graph = [
            [1, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 0, 0, 0]
        ]

        route_mat, dispatch_mat, obj_val = vrp(graph, dispatch_vec, capacity_vec, start_loc_vec)

        route_name_vec = getEdgeNameVec(graph)

        print('obj val: ' + str(obj_val))
        print("routes:")
        for row in route_mat:
            print(str([route_name_vec[edgeN] + ': ' + str(int(row[edgeN])) for edgeN in range(len(row))]))

        print("dispatch:")
        for row in dispatch_mat:
            print(str(row))

        assert obj_val == 6
        for row in route_mat:
            assert sum(row) == 2
        assert sum([sum(row) for row in dispatch_mat]) == sum(dispatch_vec)

        print('=======================================')
        print()
        print()
        """

if __name__ == '__main__':
    unittest.main()
