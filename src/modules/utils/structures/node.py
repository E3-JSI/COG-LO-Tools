class Node:
    """
    Encapsulates Node data from JSON object
    """
    def __init__(self, node):
        self.id = node['uuid']
        self.name = node['address']
        self.lat = node['lat']
        self.lon = node['lon']
        self.cluster = None

