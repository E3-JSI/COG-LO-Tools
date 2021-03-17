class Parcel:
    """
    Stores parcel information for single delivery
    Target is a destination Node
    Default package volume for poc is 1, for easier UUID unpacking
    """
    def __init__(self, uuid, target, volume, current_location, type=None, country=None, pickup_location=None, destination_location=None):
        self.volume = volume
        self.target = target
        self.uuid = uuid
        self.current_location = current_location
        self.pickup_location = pickup_location
        self.destination_location = destination_location
        self.type = type
        self.country = country