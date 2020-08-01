
##GRAPH_TEST_SCENARIO

#!/usr/bin/python3
import os
import math
import json
import networkx as nx
import matplotlib.pyplot as plt

from src.modules.create_graph.data_parser.data_handler import DataHandler
from src.modules.create_graph.neighbours_finder import NeighboursFinder
from src.modules.create_graph.pojo.pruneG import GraphPrune

class JsonGraphCreator():

    @staticmethod
    def graph_viz(nodes, ways):
        labeled_G = nx.Graph()
        not_labeled_G = nx.Graph()
        n = set()

        colors = [];
        colorMap = {};
        i = 0
        for edge in ways:
            l = edge.get_all_nodes()
            if l[0] not in n:
                not_labeled_G.add_node(l[0], pos=(nodes[l[0]].lon, nodes[l[0]].lat))
                labeled_G.add_node(l[0], pos=(nodes[l[0]].lon, nodes[l[0]].lat))
            if l[1] not in n:
                not_labeled_G.add_node(l[1], pos=(nodes[l[1]].lon, nodes[l[1]].lat))
                not_labeled_G.add_node(l[1], pos=(nodes[l[1]].lon, nodes[l[1]].lat))

            if nodes[l[1]].is_empty_tagged() == True or nodes[l[0]].is_empty_tagged() == True:
                labeled_G.add_edge(l[0], l[1], weight=edge.distance, edge_color='b')
            else:
                not_labeled_G.add_edge(l[0], l[1], weight=edge.distance, edge_color='b')
            n.add(l[0])
            n.add(l[1])

        return (not_labeled_G, labeled_G, colors)

    @staticmethod
    def drawGraph(t, postalNodes, postalWays):
        (not_labeled_G, labeled_G, colors) = t
        G = nx.Graph()
        D = nx.Graph()
        li = []
        labels = {}

        for posts in postalNodes:
            li.append(posts)
            G.add_node(posts, pos=(postalNodes[posts]["lat"], postalNodes[posts]["lon"]))
            labels[posts] = postalNodes[posts]["post_id"]

        tmpDifColoring = []
        for posts in postalNodes:
            tmpDifColoring.append(posts)
            D.add_node(posts, pos=(postalNodes[posts]['post'].latitude, postalNodes[posts]['post'].longitude))

        for src, dest, weight in postalWays:
            G.add_edge(src, dest, weight=weight)

            # plt.figure(1)

        # pos = nx.spring_layout(G, k=0.25, iterations=40)
        pos = nx.get_node_attributes(not_labeled_G, 'pos', )
        # pos = nx.get_node_attributes(not_labeled_G, 'pos', )
        nx.draw(not_labeled_G, pos=pos, node_size=0.1, node_color='g', edge_color='g', )
        nx.draw(labeled_G, pos=pos, node_size=0.0, edge_size=1, node_color='r', edge_color='r')
        nx.draw_networkx_nodes(G, pos, nodelist=tmpDifColoring, node_color='g')
        if len(postalWays) != 0:
            nx.draw_networkx_nodes(D, pos, nodelist=li, node_color='b')
            col = nx.draw_networkx_edges(G, pos, nodelist=li, node_color='g', edge_color='r')
            col.set_zorder(20)
            nx.draw_networkx_labels(G, pos, labels, font_size=16)
        plt.show()

    @staticmethod
    def drawStaticGraph(nodes, ways, results):
        G = nx.Graph()
        li = []

        #    edge
        labels = {}
        i = 0
        color_map = []
        for posts in nodes:
            if nodes[posts].post_id != None:
                if nodes[posts].post_id in results:
                    color_map.append('red')
                else:
                    color_map.append('blue')
                labels[i] = nodes[posts].post_id + "(" + str(posts) + ")"
            else:
                labels[i] = posts
                color_map.append('green')

            li.append(posts)
            G.add_node(posts)

            i = i + 1

        for key, value in ways.items():
            for kc, vc in value.items():
                G.add_edge(key, kc, weight=vc['weight'])

        import matplotlib.pyplot as plt

        # plt.figure(1)
        pos = nx.spring_layout(G, k=0.1, iterations=80)
        nx.draw_networkx_nodes(G, pos, node_color=color_map, nodelist=li)
        nx.draw_networkx_edges(G, pos, nodelist=li, node_color='g')

        nx.draw_networkx_labels(G, pos, labels, font_size=16)

        plt.show()

    def create_json_graph(self, config_path):
        with open(config_path) as json_data:
            json_config = json.load(json_data)

        import time
        start_time = time.time()
        osmHandler = DataHandler(
            json_config["map_basic"],
            {json_config["post_loc_type"]: json_config["post_loc"]}
        )
        print("Pre-step  {}".format(time.time() - start_time))

        # G = osmHandler.graph_viz()
        roadNodes = osmHandler.modified_nodes
        roadWays = osmHandler.modified_ways
        ways = osmHandler.ways
        map_posts_to_nodes = {}
        for k, v in roadNodes.items():
            if v.is_post:
                map_posts_to_nodes[v.post_id] = k
                # map_posts_to_nodes[v.post.uuid] = k
                print("Post_id: " + str(v.post_id) + " node_id" + str(k))

        postNode = {}
        postNodePlain = {}
        postEdge = set()
        tmpRes = []

        finder = NeighboursFinder(None)

        for postId, nodeId in map_posts_to_nodes.items():

            res = finder.search_near_posts(roadNodes, roadWays, ways, nodeId, map_posts_to_nodes, json_config["eps"])
            print('PostID ' + str(postId) + ' Node: ' + str(nodeId) + ' r: ' + str(res))

            tmpRes.append((postId, nodeId, res))
            for res_id, res_dist in res:
                dist = math.floor(res_dist * 1000)
                postEdge.add((nodeId, map_posts_to_nodes[res_id], dist))
                postEdge.add((map_posts_to_nodes[res_id], nodeId, dist))
                # print(res_id)

            for k, v in roadNodes.items():
                if v.post_id != None:
                    d = {'address': v.address,
                         'lat': v.lat,
                         'lon': v.lon,
                         'uuid': v.post.uuid,
                         # 'node_id': v.node_id,
                         # 'post_id': v.post_id
                         }
                    postNode[k] = v.__dict__
                    postNodePlain[k] = d
        # print final graph
        if len(postEdge) != 0:
            (not_labeled_G, labeled_G, colors) = self.graph_viz(roadNodes, ways)
            self.drawGraph((not_labeled_G, labeled_G, colors), postNode, postEdge)

        # prune graph
        graph = {'nodes': postNodePlain, 'edge': list(postEdge)}
        graph2 = GraphPrune().PruneG(graph)
        f = open("Graph_final.json", "w")
        f.write(json.dumps(graph2))
        f.close()

        '''
        nodes_dict, edges_dict = syntetic_graph1_construction()
        import modules.create_graph.neighbours_finder as neighbour_alg
        finder = neighbour_alg.NeighboursFinder()
        result = finder.search_near_posts(nodes_dict, edges_dict, 0, 1)
        print(result)
        '''


if __name__ == "__main__":
    creator = JsonGraphCreator()
    #os.system("dir")
    creator.create_json_graph('./create_graph/config/config1.json')