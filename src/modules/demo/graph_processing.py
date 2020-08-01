import json
from queue import PriorityQueue
from math import sin, cos, sqrt, atan2, radians, inf

from ..utils.structures.node import Node
from ..utils.structures.edge import Edge
from ..utils.structures.path import Path


@DeprecationWarning
class GraphLoader:
    """Loads graph data from static JSON file"""

    def __init__(self, path='modules/demo/data/posts.json'):
        self.path = path
        self.nodes, self.edges = self._load_graph()

    def _load_graph(self):
        """Loader method that constructs lists of graph nodes and edges"""
        with open(self.path, "r", encoding='UTF-8') as read_file:
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

        return nodes, edges


class GraphProcessor:
    """
    GraphProcessor handles path management and basic graph operations
    """

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.edge_map = self._map_edges()
        # self.paths = self._calculate_shortest_paths()
        self.incident_matrix = []
        self.make_matrix()
        print('Loaded graph with', len(nodes), 'nodes,', len(self.edges), 'edges')

    def _map_edges(self):
        """Create dict mapping lists of edges to starting nodes for faster later search"""
        edge_map = {}
        for node in self.nodes:
            edge_map[node.id] = []

        for e in self.edges:
            edge_map[e.start].append(e)
        return edge_map

    @staticmethod
    def _backtrack_path(came_from, goal, start):
        """Produces a list of nodes from A* output by backtracking over nodes"""
        nodes = [goal]

        while nodes[-1] != start:  # recreate path in backward order, as sequence od nodes
            tmp = came_from[nodes[-1]]
            nodes.append(tmp)
        nodes.reverse()
        return nodes

    @staticmethod
    def print_path(path):
        """Prints a backtracked path in readable format"""
        print("{0} nodes: ".format(len(path)), end='')
        print(" -> ".join([str(n.id) for n in path]))

    def _get_neighbours(self, node):
        """Find Node's neighbours from end connection of its edges"""
        nodes = []
        for e in self.edge_map[node.id]:
            nodes.append(next((x for x in self.nodes if x.id == e.end), None))
        return nodes

    def get_cost(self, a, b):
        """Get accurate cost of single edge"""
        for e in self.edge_map[a.id]:
            if e.start == a.id and e.end == b.id:
                return e.cost

    def _find_shortest(self, start, goal):
        """Executes A* path searching.
        Returns a list of nodes that form optimal path based on start and target nodes"""

        node_queue = PriorityQueue()
        node_queue.put((0, start), 0)

        came_from = {start: None}
        cost_so_far = {start: 0}

        while not node_queue.empty():
            current = node_queue.get()[1]

            if current == goal:  # path finished
                break

            for n in self._get_neighbours(current):
                new_cost = cost_so_far[current] + self.get_cost(current, n)
                if n not in cost_so_far or new_cost < cost_so_far[n]:
                    cost_so_far[n] = new_cost
                    priority = new_cost + self.distance(goal, n)
                    node_queue.put((priority, n), priority)
                    came_from[n] = current
        nodes = self._backtrack_path(came_from, goal, start)

        return Path(nodes, cost_so_far[goal])

    def _calculate_shortest_paths(self):
        """Compute paths to all nodes from all nodes in graph. Useful for precomputing paths"""
        paths = {}
        print('Computing all paths in partition')
        for i, start in enumerate(self.nodes):
            for j, end in enumerate(self.nodes):
                path = self._find_shortest(start, end)
                if start not in paths:
                    paths[start] = []
                paths[start].append(path)
        return paths

    def arbitrary_distance(self, lat1, lon1, lat2, lon2):
        """Distance between 2 geo points"""
        return self.__distance(lat1, lon1, lat2, lon2)

    def distance(self, a, b):
        """Distance between Node objects"""
        return self.__distance(a.lat, a.lon, b.lat, b.lon)

    @staticmethod
    def __distance(latitude1, longitude1, latitude2, longitude2):
        earth_radius = 6373.0

        lat1 = radians(latitude1)
        lon1 = radians(longitude1)
        lat2 = radians(latitude2)
        lon2 = radians(longitude2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = earth_radius * c
        return distance

    def node_from_id(self, uuid):
        """Finds a node from its name/uuid"""
        for n in self.nodes:
            if n.id == uuid:
                return n

    def get_path(self, a, b):
        """finds a path from Node A to Node B"""
        return self._find_shortest(a, b)

    def make_matrix(self):
        """Build incidence matrix for graph"""
        self.incident_matrix = []

        for ni, n in enumerate(self.nodes):
            tmp_arr = [0] * len(self.edges)
            for ne, e in enumerate(self.edges):
                if e.start == n.id:
                    tmp_arr[ne] = 1
            self.incident_matrix.append(tmp_arr)

    def map_vehicles(self, vehicles):
        """Map vehicles to closest node for demo code"""
        map_trucks = []
        for truck in vehicles:
            min_dist = inf
            node = None
            for node in self.nodes:
                dist = self.__distance(truck['latitude'], truck['longitude'], node.lat, node.lon)
                if dist < min_dist:
                    min_dist = dist
                    node = node
            map_trucks.append(node)
        return map_trucks
