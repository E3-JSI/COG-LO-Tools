import json
import os
import pickle

from ..create_graph.config.config_parser import ConfigParser
from ..create_graph.pojo.pruneG import GraphPrune
from ..partitioning.post_partitioning import GraphPartitioner

config_parser = ConfigParser()


class GraphPreprocessing:
    """
    Class used for pre-processing graphs based on use_case defined.
    """
    @staticmethod
    def extract_graph_processors(use_case):
        """
        Method used for pre-processing graphs. Argument given is 'use_case' upon
        which we decide whether graph parsing needs to be implemented.
        Method returns graph processors.
        :param use_case:
        :return:
        """
        graph_path = config_parser.get_graph_path(use_case)
        pickle_path = config_parser.get_pickle_path(use_case)
        partitioner = GraphPreprocessing.init_partitioner(graph_path, pickle_path)

        return partitioner.graphProcessors

    @staticmethod
    def save_graph_file(nodes, edges, graph_path):
        graph = {'nodes': nodes, 'edge': edges}
        graph_prune = GraphPrune().PruneG(graph)
        graph_file = open(graph_path, "w")
        graph_file.write(json.dumps(graph_prune))
        graph_file.close()

    @staticmethod
    def init_partitioner(graph_path, pickle_path):
        """
        Method used for loading Partitioner instance from pickle file or initializing
        new partitioner and returning it's instance.
        :param use_case: use_case can be SLO-CRO or ELTA for now.
        :return: instance of GraphPartitioner
        """
        # Load locally stored pickle
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as loadfile:
                partitioner = pickle.load(loadfile)
            print('Loaded pickled graph partitioner data')
        else:
            # Initialize new GraphPartitioner based on use_case given as parameter and execute partition() method,
            # then store partitioner instance in a pickle file and return it
            print('No data found, runing init GraphPartitioner and partition procedure')
            partitioner = GraphPartitioner(graph_path)
            partitioner.partition(config_parser.get_graph_partitions())
            with open(pickle_path, 'wb') as dumpfile:
                pickle.dump(partitioner, dumpfile)
                print('Stored pickled dump for future use')

        return partitioner




