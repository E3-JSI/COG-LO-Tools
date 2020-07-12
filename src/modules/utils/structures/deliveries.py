class Deliveries:
    # Stores total parcel demand information, before, new, and final_total
    def __init__(self, deliveries_origin, deliveries_req, deliveries):
        self.origin = deliveries_origin
        self.req = deliveries_req
        self.deliveries = deliveries
