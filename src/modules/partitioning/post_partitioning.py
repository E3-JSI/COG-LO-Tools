import copy
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..create_graph.config.config_parser import ConfigParser
from ..demo.graph_processing import GraphProcessor
from ..partitioning.spectral import spectral_part
from ..utils.structures.edge import Edge
from ..utils.structures.node import Node

config_parser = ConfigParser()


class ClusterRender:
    """Renders partitioned points in chart"""

    def __init__(self):
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:cyan', 'tab:pink', 'tab:purple']

    def render(self, nodes, assignment):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(nodes)):
            clustN = int(assignment[i])
            node = nodes[i]
            clr = self.colors[clustN]
            ax.scatter(node.lat, node.lon, c=clr)

        plt.show()


class GraphPartitioner:
    """Handles partitioning of graph from static JSON files using spectral partitioning"""

    def __init__(self, graph_path):
        self.path = graph_path
        self.nodes = []
        self.edges = []
        self.graphProcessors = []
        self._load_graph(self.path)
        self.matrix = []
        self.cluster_renderer = ClusterRender()
        self.adjacency_matrix()

    def _load_graph(self, path):
        os.system("ls")
        """Load graph from file"""
        with open(path, "r", encoding='UTF-8') as read_file:
            data = json.load(read_file)
            nodes = []
            edge_map = {}
            for n in data['nodes']:
                node = Node(data['nodes'][n])
                nodes.append(node)
                edge_map[node.id] = []

            edges = []
            for e in data["edge"]:
                edge = Edge(e, data["nodes"])
                edge_map[edge.start].append(edge)
                edges.append(edge)

        self.edges = edges
        self.nodes = nodes

    def adjacency_matrix(self):
        """Compute adjacency matrix"""
        n = len(self.nodes)
        self.matrix = np.zeros((n, n))
        node_map = {}
        for i in range(len(self.nodes)):
            node_map[self.nodes[i].id] = i

        for i in range(len(self.edges)):
            e = self.edges[i]
            idxA = node_map[e.start]
            idxB = node_map[e.end]
            self.matrix[idxA, idxB] += 1

    def partition(self, count):
        """Prepare matrices, partition graph and assign points and edges to computed partitions"""
        print('Running partitioning for', count, 'partitions on', len(self.nodes), 'nodes')
        if count == 1:
            self.graphProcessors = [GraphProcessor(self.nodes, self.edges)]
            print('Only one partition made', "nodes:", len(self.nodes), "edges:", len(self.edges))
            return self.graphProcessors

        start_time = time.time()
        A = self.matrix

        n_parts = count
        A = A - np.diag(np.diag(A))

        # partition graph
        assignments, Y = spectral_part(A, n_parts)

        # plot graphs
        print("Partitioning took {:3f}".format(time.time() - start_time))

        # each element of array is a single partition
        node_partitions = [[] for _ in range(n_parts)]
        edge_partitions = [set() for _ in range(n_parts)]

        #plot clusters
        #self.cluster_renderer.render(self.nodes, assignments)

        # assign nodes to partitions
        for i in range(len(self.nodes)):
            clustN = int(assignments[i])
            node = self.nodes[i]
            node.cluster = clustN
            node_partitions[clustN].append(node)

        assert sum([len(x) for x in node_partitions]) == len(self.nodes)
        print('Partitions of size: ', [len(x) for x in node_partitions])
        print('Processing', len(self.edges), 'edges')

        # assign edges to partitions
        # this stuff should be done with dicts or more efficient data structures
        copy_edges = self.edges.copy()
        # check all partitions
        for i, partition in enumerate(node_partitions):
            # check all remaining edges
            for edge in tqdm(copy_edges):
                start = edge.start
                end = edge.end
                added = False
                k = 0
                # check all partitions until edge is not added to one of them
                while k < len(partition) and not added:
                    n = partition[k]
                    if n.id == start:
                        start = None
                    elif n.id == end:
                        end = None
                    # if we found both start and end node in this partition,
                    # add edge and reversed edge (graph is undirected)
                    if start is None and end is None:
                        edge_partitions[i].add(edge)
                        reversed = copy.deepcopy(edge)
                        reversed.start = edge.end
                        reversed.end = edge.start
                        edge_partitions[i].add(reversed)
                        added = True
                    k += 1
            # delete all newly assigned edges from global edge list, to shorten checking
            for item in edge_partitions[i]:
                if item in copy_edges:
                    copy_edges.remove(item)

        # set to list, so later operations can use list operations
        edge_partitions = [list(x) for x in edge_partitions]
        self.graphProcessors = [GraphProcessor(node_partitions[i], edge_partitions[i]) for i in range(count)]
        print('Input', len(self.edges), 'assigned', sum([len(x) for x in edge_partitions]))

        return self.graphProcessors

if __name__ == '__main__':
    graph = GraphPartitioner(config_parser.get_slo_graph_path())
    node_parts, edge_parts = graph.partition(5)
