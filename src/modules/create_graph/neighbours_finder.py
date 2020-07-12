import copy
import math
import time

import matplotlib.pyplot as plt
import networkx as nx
from .pojo.front_data import FrontData

class NeighboursFinder():

    def __init__(self, G=None):
        self.G = G

    def graph_viz(self, nodes, way, id):
        labeled_G = nx.Graph()
        not_labeled_G = nx.Graph()
        n = set()

        colors = [];
        colorMap = {};
        i = 0
        for edge in way:
            l = edge.get_all_nodes()
            if l[0] not in n:
                not_labeled_G.add_node(l[0], pos=(nodes[l[0]].lon, nodes[l[0]].lat))
                labeled_G.add_node(l[0], pos=(nodes[l[0]].lon, nodes[l[0]].lat))
            if l[1] not in n:
                not_labeled_G.add_node(l[1], pos=(nodes[l[1]].lon, nodes[l[1]].lat))
                not_labeled_G.add_node(l[1], pos=(nodes[l[1]].lon, nodes[l[1]].lat))

            if (nodes[l[1]].is_empty_tagged() == True or nodes[l[0]].is_empty_tagged() == True) and (nodes[l[0]].getTagId() == id or nodes[l[1]].getTagId() == id ):
                labeled_G.add_edge(l[0], l[1], weight=edge.distance, edge_color='b')
            else:
                not_labeled_G.add_edge(l[0], l[1], weight=edge.distance, edge_color='b')
            n.add(l[0])
            n.add(l[1])

        return (not_labeled_G, labeled_G, colors)

    def drawGraph(self, t, map_posts_to_nodes):
        (not_labeled_G, labeled_G, colors) = t
        G = nx.Graph()
        D = nx.Graph()
        li = []
        labels = {}
        '''
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
        '''
        # pos = nx.spring_layout(G, k=0.25, iterations=40)
        pos = nx.get_node_attributes(not_labeled_G, 'pos', )
        # pos = nx.get_node_attributes(not_labeled_G, 'pos', )
        nx.draw(not_labeled_G, pos=pos, node_size=0.1, node_color='g', edge_color='g', )
        nx.draw(labeled_G, pos=pos, node_size=0.0, edge_size=1, node_color='r', edge_color='r')
        '''
        nx.draw_networkx_nodes(G, pos, nodelist=tmpDifColoring, node_color='g')
        if len(postalWays) != 0:
            nx.draw_networkx_nodes(D, pos, nodelist=li, node_color='b')
            col = nx.draw_networkx_edges(G, pos, nodelist=li, node_color='g', edge_color='r')
            col.set_zorder(20)
            nx.draw_networkx_labels(G, pos, labels, font_size=16)
        '''
        plt.show()
    '''
    def drawGraph(self, H_roads, front, postsRes, node_id_node_map_tmp):
        plt.clf()
        plt.cla()
        plt.close()

        G_front = nx.Graph()
        K_posts = nx.Graph()
        li = []
        for k, v in front.items():
            li.append(k)
            G_front.add_node(k, pos=(node_id_node_map_tmp[k].lat, node_id_node_map_tmp[k].lon))

        color_map = []
        posts = []
        if len(postsRes) != 0:
            for k, v in node_id_node_map_tmp.items():
                if v.post_id != None and not (v.post_id in postsRes[0]):
                    posts.append(k)
                    K_posts.add_node(k, pos=(node_id_node_map_tmp[k].lat, node_id_node_map_tmp[k].lon))
                    color_map.append('blue')
                elif v.post_id != None and v.post_id in postsRes[0]:
                    posts.append(k)
                    K_posts.add_node(k, pos=(node_id_node_map_tmp[k].lat, node_id_node_map_tmp[k].lon))
                    color_map.append('red')

        # pos = nx.spring_layout(G, k=0.25, iterations=40)
        pos = nx.get_node_attributes(H_roads, 'pos', )

        nx.draw(H_roads, pos=pos, node_size=0.1 )
        nx.draw_networkx_nodes(G_front, pos, node_size=100, nodelist=li, node_color='rosybrown')
        nx.draw_networkx_nodes(K_posts, pos, nodelist=posts, node_color=color_map)

        plt.show()
        G_front.clear()
        K_posts.clear()
        # H_roads.clear()
    '''
    #return self.__second_step_alg(node_id_node_map_tmp, node_id_edge_map, origin_node_id)
    def __second_step_alg(self, node_id_node_map_temp, node_id_edge_map, origin_node_id):
        front = {
            origin_node_id: FrontData(0, [] )
        }
        node_id_node_map = copy.deepcopy(node_id_node_map_temp)
        node_id_edge_map = copy.deepcopy(node_id_edge_map)


        visited_node_ids = set()
        results = []

        visited_node_ids.add(origin_node_id)
        start_time = time.time()
        exhausted_paths_count = 0

        # the counter is increased when have aleready one post office and than increase to the 3
        # if there are already two or more we are not touch it
        # also we have to decrese counter only if it has more or exactly 2 post offices in list
        counter_added = 0
        counter_modified = 0
        counter_snapped = 0
        counter_deleted = 0

        while len(front) != 0:

            active_node_id = None
            prev_node_id = None
            current_visited_post_ids = None  # TODO: do we need this in the loop??
            min_distance = math.inf

            # The FRONT: this loop finds the node that is closest to the front
            for node_id, node_data in front.items():
                # print(node_id)
                # print(node_data)

                if node_id not in node_id_edge_map:
                    continue

                node_dist = node_data.origin_dist
                neigbours = node_id_edge_map[node_id]  # get neighbors od te tocke

                for neigbour in neigbours:
                    edge_dist = neigbours[neigbour]["weight"]
                    if ((node_dist + edge_dist) < min_distance) and (neigbour not in visited_node_ids):
                        min_distance = node_dist + edge_dist
                        active_node_id = neigbour
                        current_visited_post_ids = [val for val in node_data.prev_posts]
                        prev_node_id = node_id

            # print(active_node_id)
            # print(min_distance)
            # print(current_visited_points)
            # print(prev_node_id)
            #print("Runtime second step, finding the neighbour: {}".format(time.time() - start_time))
            if active_node_id is None:
                # we were unable to find any new neighbours to extend the front
                # that means that we have searched through the entire graph
                # we can safely exit the loop
                print('Cannot extend the front. No neigbours available. Finishing search.')
                print(str(front))
                break
            #print('active node: ' + str(active_node_id))

            if active_node_id not in node_id_node_map:
                raise ValueError('Found an invalid node: ' + str(active_node_id))

            # check if the active node is a post office. if so, add it to the results

             #is_post = node_id_node_map[active_node_id].post_id is not None      ##check the
             #is_post = active_node_id.isTagged is not None  ##check the
             #if is_post:
             #   current_visited_post_ids += [active_node_id]
             #   if len(current_visited_post_ids) == 1:
             #       results += [(node_id_node_map[active_node_id].post_id, min_distance)]
            # check if the active node is a post office. if so, add it to the results
            is_post = node_id_node_map[active_node_id].is_empty_tagged()
            if is_post and not  node_id_node_map[active_node_id].getTagId() == origin_node_id:
                is_found = False
                for i in range (len(current_visited_post_ids)):
                    current_post_id = node_id_node_map[active_node_id].getTagId()
                    if node_id_node_map[current_visited_post_ids[i]].getTagId() == current_post_id:
                        is_found = True
                if not is_found:
                    current_visited_post_ids += [active_node_id]
                    if len(current_visited_post_ids) == 1:
                        tagid = node_id_node_map[active_node_id].getTagId()
                        results += [(node_id_node_map[tagid].post_id, min_distance)]
            #print("Runtime second step, check if post on node id & writte results: {}".format(time.time() - start_time))
            # add the active node to the list of visited nodes
            visited_node_ids.add(active_node_id)

            # only add the active node to the front if it has any
            # non-visited neighbours
            active_node_neigh_ids = node_id_edge_map[active_node_id]
            all_neighbours_visited = True
            for neigh_id in active_node_neigh_ids:
                if neigh_id not in visited_node_ids:
                    all_neighbours_visited = False
                    break

            # if all the neighbours of the active node are visited, we need
            # to check if we can delete them now after one of their neighbours
            # (the active node) has been visited - Handling the use case when cycle/loop has been reached for two edges

            for neigh_id in active_node_neigh_ids:
                if neigh_id == prev_node_id or neigh_id not in front:
                    continue
                all_neigh_of_neigh_visited = True
                neigh_of_neigh_ids = node_id_edge_map[neigh_id]
                for neigh_of_neigh_id in neigh_of_neigh_ids:
                    if neigh_of_neigh_id not in visited_node_ids:
                        all_neigh_of_neigh_visited = False
                        break
                if all_neigh_of_neigh_visited:
                    if len(front[neigh_id].prev_posts) >= 2:
                        #print('decreased cnt')
                        exhausted_paths_count -= 1
                        counter_deleted += 1
                    del front[neigh_id]
                 #   print('deleted neighbour of neighbour: ' + str(neigh_of_neigh_id))

            # adding nodes to the front
            if not all_neighbours_visited:
                # add the node to the front
                front[active_node_id] = FrontData(
                    min_distance,
                    current_visited_post_ids
                )
                #print("Added : " + str(active_node_id) + " n_posts: " + str(len(current_visited_post_ids)))
                if len(current_visited_post_ids) >= 2:
                    #print('increased cnt')
                    exhausted_paths_count += 1
                    counter_added += 1
            #print("Runtime second step, adding nodes to front : {}".format(time.time() - start_time))
            # if all the neighbours of the previous point have been visited, remove it
            # from the front
            # TODO: optimize this - have a counter in each node that counts
            # TODO: how many neighbours have been visited

            prev_node_neighbour_ids = node_id_edge_map[prev_node_id]
            all_neighbours_visited = True
            for neighbour_id in prev_node_neighbour_ids:
                if neighbour_id not in visited_node_ids:
                    all_neighbours_visited = False
                    break

            # delete the nodes that have all the neighbours visited
            if all_neighbours_visited:
                if len(front[prev_node_id].prev_posts) >= 2:
                    #print('decreased cnt')
                    exhausted_paths_count -= 1
                    counter_deleted += 1
                #print('Node deleted: ' + str(prev_node_id) + ' n_posts: ' + str(len(front[prev_node_id].prev_posts)))
                del front[prev_node_id]

            #  check if the algorithm is finished
            #print('total paths: ' + str(len(front)) + ', exhausted paths: ' + str(exhausted_paths_count))
            # print('added: ' + str(counter_added) + ' Deleted:  ' + str(counter_deleted) +
            #      ' Modified : ' + str(counter_modified) + ' Snapped : ' + str(counter_snapped))
            #      print("results:" + str(results))
            #if time.time() - start_time >= 0.08:
            #    self.drawGraph(self.G, front, results, node_id_node_map)
            #    start_time = time.time()
                #print("visited:" + str(visited_node_ids))

            # when there is more than one post office on in all direction, the search can be terminated
            if exhausted_paths_count == len(front):
                #print("All paths exhausted! Terminating the algorithm!")
                break

        print("Runtime second step: {}".format(time.time() - start_time))
        return results

    def __fist_step_alg(self, node_id_node_map, node_id_edge_map, start_node_id, origin_node_id, eps_km):
        # front = [(start_node_id, 0, [])]
        F = [(start_node_id, 0)]
        visited_ids = set()
        start_time = time.time()

        while len(F) > 0:
            current_tup = F.pop(0)
            current_id = current_tup[0]
            current_dist = current_tup[1]
            edge_map = node_id_edge_map[current_id]  # get neighbors od te tocke
            for neighbour_id in edge_map:
                current_neighbour_dist = node_id_edge_map[current_id][neighbour_id]['weight']
                if neighbour_id in visited_ids:
                    continue
                if current_dist + current_neighbour_dist > eps_km:
                    continue
                F.append((neighbour_id, current_dist + current_neighbour_dist))
            visited_ids.add(current_id)
            current_node = node_id_node_map[current_id]
            if not current_node.getTagId() ==origin_node_id:
                id, dist = current_node.tag_filter()
                if (dist is None) or (dist is not None and dist > current_dist):
                    current_node.addTag((start_node_id, current_dist))
        # Set the id of the nearest post on .tag
        #for id, value in node_id_node_map.items():
        #    node_id_node_map[id].tag = node_id_node_map[id].tag_filter()

        #print("Runtime first step: {}".format(time.time() - start_time))


    def search_near_posts(self, node_id_node_map, node_id_edge_map, ways, origin_node_id,map_posts_to_nodes, eps_km):
        for id, value in node_id_node_map.items():
            node_id_node_map[id].clean_tagged()
        self.__fist_step_alg(node_id_node_map, node_id_edge_map, origin_node_id, origin_node_id, eps_km)
        #(not_labeled_G, labeled_G, colors) = self.graph_viz(node_id_node_map, ways)
        #self.drawGraph((not_labeled_G, labeled_G, colors), map_posts_to_nodes, m)
        for node_id in map_posts_to_nodes:
            if map_posts_to_nodes[node_id]!= origin_node_id:
                self.__fist_step_alg(node_id_node_map, node_id_edge_map, map_posts_to_nodes[node_id], origin_node_id, eps_km)

        for node,obj in node_id_node_map.items():
            if obj.getTagId() is not None and obj.getTagId() == 243:
                print(obj)
        #partial graph printing
        #for node_id,key_node_id in map_posts_to_nodes.items():
         # (not_labeled_G, labeled_G, colors) = self.graph_viz(node_id_node_map, ways, key_node_id)
          #self.drawGraph((not_labeled_G, labeled_G, colors), map_posts_to_nodes)
        return self.__second_step_alg(node_id_node_map, node_id_edge_map, origin_node_id)
