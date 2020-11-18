"""Class OrderRelathions handles conflist routing multiple different pickup locations nodes
are in conflict, creating ordering loop cycles: different nodes have parcels with same pickup locaiton"""
import collections
import copy

class DagNode:

    def __init__(self, node_id, src_id_set):
        self._node_id = node_id
        self._src_id_set = src_id_set

    def get_node_id(self):
        return self._node_id

    def has_predecessors(self):
        return len(self._src_id_set) > 0

    def remove_src_node(self, node):
        src_id_set = self._src_id_set

        remove_node_id = node.get_node_id()
        if remove_node_id in src_id_set:
            src_id_set.remove(remove_node_id)

    def __lt__(self, other):
        return self._node_id in other._src_id_set

    def __str__(self):
        return str(self._node_id) + ': ' + str(self._src_id_set)


class OrderRelations:

    @staticmethod
    def create_relations(PickupNodes, route):
        #if no psarcels to deliver
        #if (len(PickupNodes) == 1):
        #    return (list(PickupNodes)[0])

        relations = []
        # start node has all dependencies, as needs to be pushed to beginning
        nodesKeys = [key for key, value in PickupNodes.items()]
        for station in route[1:]:
            relations.append((nodesKeys[0], station["id"]))
        # relations for pickup locations
        for key in nodesKeys[1:]:
            load_parcels = PickupNodes[key]["load"]
            for station in route:
                for parcel in load_parcels:
                    if parcel in station["unload"]:
                        relations.append((key, station["id"]))
        relations_ordered = OrderRelations.order_relations(relations)

        # creating new stations for double visits with two separate loading and unloading location
        duplicates_load = [item for item, count in collections.Counter(relations_ordered).items() if count > 1]
        duplicates_unload = [item for item, count in collections.Counter(relations_ordered).items() if count > 1]

        stations = {}
        for station in route:
            stations[station["id"]] = station
        stations_ordered = []
        for item in relations_ordered:
            if item not in duplicates_load and item not in duplicates_unload:
                stations_ordered.append(stations[item])
            elif item in duplicates_load and item in duplicates_unload:
                station = copy.deepcopy(stations[item])
                station["unload"] = []
                stations_ordered.append(station)
                duplicates_load.remove(item)
            elif item not in duplicates_load and item in duplicates_unload:
                station = copy.deepcopy(stations[item])
                station["load"] = []
                stations_ordered.append(station)

        return stations_ordered

    @staticmethod
    def topological_sort(node_list):
        sorted_list = []

        # copy the node_list to avoid changing input arguments (it's bad practice to do so)
        node_list = [node for node in node_list]

        while len(node_list) > 0:
            curr_nodeN = -1

            # find a node that has no predecessor
            for nodeN, node in enumerate(node_list):
                if not node.has_predecessors():
                    curr_node = node
                    curr_nodeN = nodeN
                    break

            if curr_nodeN < 0:
                raise ValueError('Failed to find a node with no predecessor! Are you sure you have a DAG?')

            curr_node = node_list.pop(curr_nodeN)
            sorted_list.append(curr_node)

            # remove the current node as source from all remaining nodes
            for node in node_list:
                node.remove_src_node(curr_node)

        return sorted_list

    @staticmethod
    def order_relations(relations):
        internal2external_id_map = {}

        for src_id, dst_id in relations:
            internal2external_id_map[src_id] = src_id
            internal2external_id_map[dst_id] = dst_id

        # create a list of non-cyclic relations by duplicating
        # some nodes
        dag_relations = []

        used_relation_set = set()
        new_dst_id_map = {}
        for src_id, dst_id in relations:
            if (dst_id, src_id) in used_relation_set:
                # check if a new destination id already exists
                if dst_id not in new_dst_id_map:
                    # instead of i -> j add the relation i -> j'
                    new_dst_id = len(internal2external_id_map)
                    new_dst_id_map[dst_id] = new_dst_id
                    internal2external_id_map[new_dst_id] = dst_id

                new_dst_id = new_dst_id_map[dst_id]
                dag_relations.append((src_id, new_dst_id))
            else:
                dag_relations.append((src_id, dst_id))

            used_relation_set.add((src_id, dst_id))

        # topologically sort the generated list of non-cyclic relations (DAG)
        node_id_set = set()
        for src_id, dst_id in dag_relations:
            node_id_set.add(src_id)
            node_id_set.add(dst_id)

        node_list = []

        node_src_map = {node_id: set() for node_id in node_id_set}

        for src_id, dst_id in dag_relations:
            node_src_map[dst_id].add(src_id)

        for node_id in node_id_set:
            node_src_id_set = node_src_map[node_id]
            node_list.append(DagNode(node_id, node_src_id_set))

        ordered_node_list = OrderRelations.topological_sort(node_list)
        node_id_vec = [internal2external_id_map[node.get_node_id()] for node in ordered_node_list]

        return node_id_vec

"""
if __name__ == '__main__':
    # (_, _, _), 4, 2, (_, _, _, _), 5, 2, 6, _, 9
    relations = [
        ('a', 'b'),
        ('b', 'a'),
        ('a', 'c'),
        ('c', 'a'),
        ('a', 'd'),
        ('c', 'b'),
        ('c', 'd'),
        ('d', 'c')
    ]

    # (pickup_loc, dropoff_loc)

    ordered_node_list = order_relations(relations)

    print('result:')
    for node in ordered_node_list:
        print(node)
"""