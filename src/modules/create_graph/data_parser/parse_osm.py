import xml.sax
from ..utils import utils
from .pojo.way import Way
from .pojo.node import Node

class OsmParsers(xml.sax.ContentHandler):
    '''
        OsmParsers parse OSM data to nodes and ways. Nodes are represents as dictionary in order to fast accesss to it.
        Ways are list.
    '''

    def __init__(self):
        self.nodes = {}  # nodes are save
        self.ways = []
        self.tmpWays = []
        self.tags = {}
        self.roads = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service",
                      "living_street", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
                      "motorway_junction"}
        #self.roads = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service",
        #              "living_street", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
        #              "motorway_junction"}

    def startElement(self, name, attrs):
        if name == "node":
            node = Node()
            if "id" in attrs and "lat" in attrs and "lon" in attrs:
                node.add_node(int(attrs["id"]), float(attrs["lat"]), float(attrs["lon"]))
                self.nodes[int(attrs["id"])] = node
        if name == "way":
            self.tmpWays = []
            self.tags = {}
            # for key in dict(attrs):
            #    print(key + ", " + attrs[key])
        if name == "nd":
            self.tmpWays.append(attrs["ref"])
        if name == "tag":
            self.tags[attrs["k"]] = attrs["v"]

    def endElement(self, name):
        if name == "way":
            if "highway" in self.tags and self.tags["highway"] in self.roads:
                for i in range(1, len(self.tmpWays)):
                    distance = 0
                    way = Way()
                    node1 = self.nodes[int(self.tmpWays[i - 1])]
                    node2 = self.nodes[int(self.tmpWays[i])]
                    way.add_path(int(self.tmpWays[i - 1]), int(self.tmpWays[i]))
                    distance = utils.calcDistance(node1.lat, node1.lon, node2.lat, node2.lon)
                    way.add_distance(distance)
                    self.ways.append(way)
                self.tags = {}
                self.tmpWays = {}

