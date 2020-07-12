class Path:
    """A list of nodes make a single path"""
    def __init__(self, path, cost):
        self.start = path[0]
        self.end = path[-1]
        self.path = path
        self.cost = cost
