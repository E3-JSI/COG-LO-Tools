class Vehicle:
    """
    This class stores information about single vehicle
    name is usually uuid
    start_node is name/matching UUID
    """
    def __init__(self, name, start_node, parcels, capacity=200):
        self.capacity = capacity
        self.name = name
        self.start_node = start_node
        self.parcels = parcels

