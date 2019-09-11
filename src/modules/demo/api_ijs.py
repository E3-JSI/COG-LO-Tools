import time
from math import inf

from flask import Flask, request
from flask_jsonpify import jsonify
from flask_restful import Resource, Api

from modules.cvrp.vrp import VRP
from modules.demo.mockup_graph import MockupGraph

SIOT_URL = 'http://151.97.13.227:8080/SIOT-war/SIoT/Server/'

brokenVehicle = None
parcelData = None
routeCache = []

class GraphProcessor:
    def __init__(self, path='modules/demo/data/9node.json'):
        self.g = MockupGraph(path)

    def map_vehicles(self, data):
        vehicles = []
        for v in data:
            loc = v["currlocation"]["currentPosition"]
            loc = loc.split(',')

            vehicles.append(
                {"id": v["UUID"],
                 "latitude": float(loc[1]),
                 "longitude": float(loc[0])})
        print("Mapping vehicles to posts")
        return self.g.map_vehicles(vehicles)

    def get_graph(self):
        return self.g.nodes, self.g.edges, self.g.incident_matrix


class VrpProcessor:
    def __init__(self, graphProcessor):
        self.vrp = VRP()
        self.graphProcessor = graphProcessor

    def process(self, metadata):
        global brokenVehicle
        if len(metadata) == 0:
            return []
        nodes, edges, graph = self.graphProcessor.get_graph()

        post_mapping = self.graphProcessor.map_vehicles(metadata)
        print("Got broken vehicle: {}".format(brokenVehicle))
        dropoff = [0] * len(nodes)
        non_broken_vehicles = []
        for i, v in enumerate(metadata):
            for loc in v['dropOffLocations']:
                target = self.graphProcessor.g.node_from_id(loc['locationId'])

                idx = nodes.index(target)
                dropoff[idx] += loc['dropoffWeightKg']
            if v['UUID'] != brokenVehicle:
                non_broken_vehicles.append(v)
            else:
                del post_mapping[i]

        start = [nodes.index(x) for x in post_mapping]
        capacity = [int(x["metadata"]["capacityKg"]) for x in non_broken_vehicles]
        # generate random load demands

        costs = [e.cost for e in edges]
        brokenVehicle = None
        routes, dispatch, objc = self.vrp.vrp(graph, dropoff, capacity, start, costs)

        return self.make_route(routes, dispatch, nodes, edges, non_broken_vehicles)

    def find_closest_post(self, loads, start, nodes):
        min_dist = inf
        min_idx = -1
        for post_idx, load in enumerate(loads):
            if load > 0:
                cur_dist = self.graphProcessor.g.distance(start, nodes[post_idx])
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_idx = post_idx
        return min_idx

    def make_route(self, graph_routes, loads, nodes, edges, vehicles):
        print("Building route from VRP output...")
        print(len(edges))
        start_nodes = self.graphProcessor.map_vehicles(vehicles)
        start_time = time.time()
        routes = []
        converted_routes = []

        for i, vehicle_load in enumerate(loads):
            start_node = start_nodes[i]
            route = [start_node]
            current_node = start_node
            vehicle_load[nodes.index(current_node)] -= vehicle_load[nodes.index(current_node)]
            cost_astar = 0
            original_vehicle_load = vehicle_load.copy()

            # always find closest post with parcels to pick/drop off.
            # start at closest node
            # get route and clear up any packages on this route
            while sum(vehicle_load) > 1:  # run until all parcels have been delivered
                post_idx = self.find_closest_post(vehicle_load, current_node,
                                                  nodes)  # idx of closest post with parcel demand
                target = nodes[post_idx]  # convert idx to node object
                vehicle_load[post_idx] -= vehicle_load[post_idx]  # take/drop all parcels
                partial_path = self.graphProcessor.g.get_path(current_node,
                                                              target)  # get path from current to next dropoff node
                for node in partial_path.path:  # drop off parcels along the way to targer
                    for idx, val in enumerate(vehicle_load):
                        if val > 0 and nodes.index(node) == idx:
                            vehicle_load[idx] -= vehicle_load[idx]

                current_node = target  # we are now at new node
                cost_astar += partial_path.cost

                # merge existing and new route, avoid adding duplicate node on start of route
                route += partial_path.path if len(partial_path.path) == 1 else partial_path.path[1:]

            # debug info
            print("Vehicle: {}".format(vehicles[i]['UUID']))
            print("Edges: VRP: {}, A*:{}".format(sum(graph_routes[i]), len(route)))

            # calculate theoretical cost of all visited edges in vrp, to compare to A*
            cost_vrp = sum([count * edges[j].cost for j, count in enumerate(graph_routes[i])])

            print("Cost VRP: {}, Cost A*:{}".format(cost_vrp, cost_astar))
            self.graphProcessor.g.print_path(route)
            print([item if x > 0 else -1 for item, x in enumerate(original_vehicle_load)])
            print(original_vehicle_load)

            routes.append(route)
            converted_routes.append({
                "UUID": vehicles[i]["UUID"],
                "route": self.route_to_sumo_format(route, original_vehicle_load, nodes)})

        print("Route build took: {}s".format(time.time() - start_time))
        return converted_routes

    def route_to_sumo_format(self, route, loads, nodes):
        converted_route = []

        if len(route) == 1 and loads[nodes.index(route[0])] == 0:
            return []

        for idx, node in enumerate(route):
            node_idx = nodes.index(node)

            converted_route.append({
                "locationId": node.id,
                "dropoffWeightKg": int(loads[node_idx]),
                "dropoffVolumeM3": int(loads[node_idx] / 10),
                "info": "This parcel must be delivered to location {0}".format(node.id),
                "position": "{},{}".format(node.lon, node.lat)
            })

        return converted_route


generalVrpProcessor = VrpProcessor(graphProcessor=GraphProcessor())
xborderVrpA3 = VrpProcessor(graphProcessor=GraphProcessor('modules/demo/data/9node_A3.json'))
xborderVrpA6 = VrpProcessor(graphProcessor=GraphProcessor('modules/demo/data/9node_A6.json'))


class Event(Resource):

    def get(self):
        return jsonify({"success": True, "message": "Please use POST request"})

    def post(self):

        global brokenVehicle, parcelData
        json = request.get_json(force=True)
        event = json['event']

        print(json)

        print("Posting to SIOT 1")
        if "broken" in event["type"].lower():
            vehicle = json['vehicle']
            brokenVehicle = vehicle['UUID']
        elif "parcel" in event['type'].lower():
            parcelData = json

        return jsonify({
            "success": True,
            "message": "Processing event for vehicle {}".format(vehicle['UUID'])
        })

class Routes(Resource):
    def get(self):
        global routeCache
        return jsonify(routeCache)

class RecReq(Resource):

    def get(self):
        return jsonify({"success": True, "message": "Please use POST request"})

    def post(self):
        data = request.get_json(force=True)
        global brokenVehicle, routeCache
        print(data)

        vehicle_metadata = data['CLOS']
        routes = []
        if "event" in data:
            if "broken" in data["event"]["type"].lower():
                brokenVehicle = vehicle_metadata[0]["UUID"]
                print("Processing VRP data.")
                routes = generalVrpProcessor.process(vehicle_metadata)
            elif "parcel" in data["event"]["type"].lower():
                print("Processing VRP data.")
                routes = generalVrpProcessor.process(vehicle_metadata)
            elif "xborder" in data["event"]["type"].lower():
                xvehiclesA3 = []
                xvehiclesA6 = []
                for v in vehicle_metadata:
                    if v["UUID"] in ["truck5F0", "truck6F0", "truck7F0", "truck8F0"]:
                        xvehiclesA3.append(v)
                    else:
                        xvehiclesA6.append(v)
                print("Running VRP 1")
                routes1 = xborderVrpA3.process(xvehiclesA3)
                print("Running VRP 2")
                routes2 = xborderVrpA6.process(xvehiclesA6)
                routes = routes1 + routes2
        if len(vehicle_metadata) < 1:
            return jsonify({"msg": "No vehicles"})

        routeCache = routes
        return jsonify({"CLOS": routes})


class CognitiveAdvisorAPI:
    def __init__(self, port=5000):
        self._port = port
        self._app = Flask(__name__)
        self._api = Api(self._app)
        self._add_endpoints()

    def _add_endpoints(self):
        self._register_endpoint('/api/adhoc/recommendationRequest', RecReq)
        self._register_endpoint('/api/adhoc/newEvent', Event)
        self._register_endpoint('/api/adhoc/getRoutes', Routes)

    def _register_endpoint(self, endpoint_name, class_ref):
        self._api.add_resource(class_ref, endpoint_name)

    def serve(self):
        self._app.run(host='0.0.0.0', port=self._port)


if __name__ == '__main__':
    server = CognitiveAdvisorAPI()

    server.serve()
