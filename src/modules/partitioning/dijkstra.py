import numpy as np

def run_dijkstra(D, startN):
    '''
    The Dijkstra algorithm to compute the shortest path to every
    node in the graph from the starting node `startN`.

    D_ij encodes the distance between nodes i and j. If there is no directo connection
         between i and j, then D_ij should be inf.
    startN is the index of the starting node
    '''
    n_nodes = D.shape[0]

    neighbour_map = []
    for srcN in range(n_nodes):
        neighbour_vec = []
        for dstN in range(n_nodes):
            if D[srcN, dstN] < np.inf and srcN != dstN:
                neighbour_vec.append(dstN)
        neighbour_map.append(neighbour_vec)


    # the vector that encodes the distances from the start
    # node to every other node
    dist_vec = np.inf*np.ones(n_nodes)

    active_node_set = set(range(n_nodes))

    dist_vec[startN] = 0
    for iterN in range(n_nodes-1):
        # select the node with the minimal distance
        activeN = -1
        active_dist = np.inf
        for candidateN in active_node_set:
            curr_dist = dist_vec[candidateN]
            if curr_dist < active_dist:
                active_dist = curr_dist
                activeN = candidateN

        if activeN == -1:
            # we were not able to find a node to expand to
            continue

        # remove the current node from the active set 
        active_node_set.remove(activeN)

        # go through all the active neighbours of the
        # current node and update the distances
        curr_dist = dist_vec[activeN]
        neighbour_vec = neighbour_map[activeN]
        for neighbourN in neighbour_vec:
            if neighbourN not in active_node_set:
                continue
            neighbour_dist = active_dist + D[activeN, neighbourN]
            if neighbour_dist < dist_vec[neighbourN]:
                dist_vec[neighbourN] = neighbour_dist

    return dist_vec


