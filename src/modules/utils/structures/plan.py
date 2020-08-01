class Plan:
    """
    Information that are input to produce plan of routing for one partition
    vehicles in this partition, all deliveries in this partition and
    the partition object, which is a GraphProcessor object
    """
    def __init__(self, vehicles, deliveries, deliveries_req, partition):
        self.vehicles = vehicles
        self.deliveries = deliveries
        self.deliveries_req = deliveries_req
        self.partition = partition
