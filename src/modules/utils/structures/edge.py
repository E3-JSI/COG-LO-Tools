class Edge:
    """
    Edge connecting 2 nodes, with a cost of travel
    """
    def __init__(self, edge, nodes):
        self.start = nodes[str(edge[0])]['uuid']
        self.end = nodes[str(edge[1])]['uuid']
        self.cost = round(edge[2], 3)

