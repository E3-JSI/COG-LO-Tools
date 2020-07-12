import os
import json
import pickle

from ..partitioning.post_partitioning import GraphPartitioner
from ..create_graph.config.config_parser import ConfigParser
from ..utils.structures.node import Node
from ..utils.structures.edge import Edge
from ..create_graph.pojo.pruneG import GraphPrune

config_parser = ConfigParser()


class GraphPreprocessing:
    """
    Class used for pre-processing graphs based on use_case defined.
    """
    @staticmethod
    def extract_graph_processors(use_case):
        """
        Method used for pre-processing graphs. Argument given is 'use_case' upon
        which we decide whether graph parsing needs to be implemented. For SLO-CRO
        graph, we need to parse graph. Method returns graph processors.
        :param use_case:
        :return:
        """
        graph_path = config_parser.get_graph_path(use_case)
        if use_case == "ELTA":
            pickle_path = config_parser.get_pickle_path(use_case)
            partitioner = GraphPreprocessing.init_partitioner(graph_path, pickle_path)

            return partitioner.graphProcessors
        else:
            """
            SLO-CRO use case
            JSON graph containing nodes and edges for both SLO and CRO will be split into
            two parts: SLO and CRO. SLO contains SLO nodes + CRO borders nodes, CRO contains
            CRO nodes + SLO border nodes. After pre-processing is completed, graphs are stored
            and partitioner for each of the graphs is initialized.
            """
            slo_graph_file = config_parser.get_slo_graph_path()
            cro_graph_file = config_parser.get_cro_graph_path()
            if not os.path.exists(slo_graph_file) or not os.path.exists(cro_graph_file):
                if os.path.exists(graph_path):
                    with open(graph_path, "r", encoding='UTF-8') as read_file:
                        data = json.load(read_file)
                        slo_nodes = {}
                        slo_nodes_id_array = []
                        slo_border_nodes = config_parser.get_border_nodes_slo()
                        slo_edges = []

                        cro_nodes = {}
                        cro_nodes_id_array = []
                        cro_border_nodes = config_parser.get_border_nodes_cro()
                        cro_edges = []

                        nodes = data["nodes"]
                        for node_index in nodes:
                            node_content = nodes[node_index]
                            node = Node(node_content)

                            # All SLO nodes + CRO border nodes
                            if "S" in node.id or node.id in cro_border_nodes:
                                slo_nodes[node_index] = node_content
                                slo_nodes_id_array.append(node.id)
                            # All CRO nodes + SLO border nodes
                            if "H" in node.id or node.id in slo_border_nodes:
                                cro_nodes[node_index] = node_content
                                cro_nodes_id_array.append(node.id)

                        edges = data["edge"]
                        for element in edges:
                            edge = Edge(element, nodes)
                            # Add edges that are in JSON graph
                            if edge.start in slo_nodes_id_array and edge.end in slo_nodes_id_array:
                                slo_edges.append(element)
                            elif edge.start in cro_nodes_id_array and edge.end in cro_nodes_id_array:
                                cro_edges.append(element)

                        # Store SLO and CRO graphs.
                        GraphPreprocessing.save_graph_file(slo_nodes, slo_edges, slo_graph_file)
                        GraphPreprocessing.save_graph_file(cro_nodes, cro_edges, cro_graph_file)
                else:
                    print("Error parsing SLO-CRO json graph. File " + graph_path + " does not exist!")
                    exit(1)

            # Create graph processors and return them
            slo_pickle_path = config_parser.get_slo_pickle_path()
            cro_pickle_path = config_parser.get_cro_pickle_path()

            slo_graph_processors = GraphPreprocessing.init_partitioner(slo_graph_file, slo_pickle_path).graphProcessors
            cro_graph_processors = GraphPreprocessing.init_partitioner(cro_graph_file, cro_pickle_path).graphProcessors

            return slo_graph_processors + cro_graph_processors

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




