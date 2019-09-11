import json
from math import sin, cos, sqrt, atan2, radians, inf
from queue import PriorityQueue
import matplotlib.pyplot as plt

class Node:
    def __init__(self, node):
        self.id = node['post_id']
        self.name = node['post_id']
        self.lat = node['lat']
        self.lon = node['lon']


class Edge:
    def __init__(self, edge, nodes):
        self.start = nodes[str(edge[0])]['post_id']
        self.end = nodes[str(edge[1])]['post_id']
        self.cost = round(edge[2], 3)

class Path:
    def __init__(self, path, cost):
        self.start = path[0]
        self.end = path[-1]
        self.path = path
        self.cost = cost

class MockupGraph:

    def __init__(self, path='modules/demo/data/posts.json'):
        self.nodes, self.edges, self.edge_map = self._load_graph(path)
        self.paths = self._calculate_shortest_paths()
        print("Got {} edges, {} nodes".format(len(self.edges), len(self.nodes)))

        self.incident_matrix = []
        self.make_matrix()

    def _backtrack_path(self, came_from, goal, start):
        """Produces a list of nodes from A* output by backtracking over nodes"""
        nodes = [goal]
        while nodes[-1] != start: #recreate path in backward order, as sequence od nodes
            tmp = came_from[nodes[-1]]
            nodes.append(tmp)
        nodes.reverse()
        return nodes

    def print_path(self, path):
        print("{0} nodes: ".format(len(path)), end='')
        print(" -> ".join([str(n.id) for n in path]))

    def _get_neighbours(self, node):
        nodes = []
        for e in self.edge_map[node.id]:
            nodes.append(next((x for x in self.nodes if x.id == e.end), None))
        return nodes

    def get_cost(self, a, b):
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

            if current == goal: #path finished
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
        paths = {}
        for i, start in enumerate(self.nodes):
            for j, end in enumerate(self.nodes):
                path = self._find_shortest(start, end)
                if start not in paths:
                    paths[start] = []
                paths[start].append(path)
        return paths

    def _load_graph(self, path):
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

        return nodes, edges, edge_map

    def distance(self, a, b):
        return self.__distance(a.lat, a.lon, a.lon, b.lon)

    def __distance(self, latitude1, longitude1, latitude2, longitude2):
        # approximate radius of earth in km
        R = 6373.0

        lat1 = radians(latitude1)
        lon1 = radians(longitude1)
        lat2 = radians(latitude2)
        lon2 = radians(longitude2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        return distance
    def node_from_id(self, id):
        for n in self.nodes:
            if n.id == id:
                return n

    def get_path(self,a,b):
        possible_paths = self.paths[a]
        for path in possible_paths:
            if path.end == b:
                return path
        print("No such path")
        return None

    def make_matrix(self):
        self.incident_matrix = []

        for ni, n in enumerate(self.nodes):
            tmp_arr = [0] * len(self.edges)
            for ne, e in enumerate(self.edges):
                if e.start == n.id or e.end == n.id:
                    tmp_arr[ne] = 1
            self.incident_matrix.append(tmp_arr)
        return 0


    def map_vehicles(self, vehicles):
        map_trucks = []
        for truck in vehicles:
            min = inf
            node = None
            for node in self.nodes:
                dist = self.__distance(truck['latitude'], truck['longitude'], node.lat, node.lon)
                if dist < min:
                    min = dist
                    node = node
            map_trucks.append(node)
        return map_trucks
