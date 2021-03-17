import time
import requests
from datetime import datetime
from math import inf
import uuid
from flask import Flask, request
from flask_jsonpify import jsonify
from flask_restful import Resource, Api
from waitress import serve
from random import randint

from ..cvrp.vrp import VRP
from ..demo.graph_processing import GraphProcessor, GraphLoader

MSB_FWD = 'http://116.203.13.198/api/postRecommendation'
brokenVehicle = None
parcelData = None

"""
This is code for Brussels demo. Should not be used for anything else!
"""

postmap = {
    "SlovenianPostpost26615": "A4",
    "SlovenianPostpost3701": "A5",
    "CroatianPostpost4617": "A7",
    "SlovenianPostpost47621": "A8",
    "SlovenianPostpost5016": "A9",
    "CroatianPostpost7674": "A10",
    "SlovenianPostpost1254": "A11",
    "CroatianPostpost3148": "A1",
    "CroatianPostpost320": "A2",
    "CroatianPostpost4398": "A6",
    "CroatianPostpost639": "A3",
    "ELTApost282": "A1",
    "ELTApost890": "A2",
    "ELTApost4647": "A3",
    "ELTApost5992": "A4",
    "ELTApost4476": "A5",
    "ELTApost1490": "A6",
    "ELTApost12188": "A7",
    "ELTApost6156": "A8",
    "ELTApost17247": "A9",
    "ELTApost30259": "A10",
    "ELTApost31073": "A11",
    "ELTApost32140": "A12",
    "ELTApost4323": "A13"
}


def uuid_by_name(name):
    for key in postmap:
        if postmap[key] == name:
            return key
    return "Unknown"


def org_by_name(name):
    if name.startswith('ELTA'):
        return "ELTA"
    elif name.startswith("Slovenia"):
        return "Slovenian Post"
    else:
        return "Croatian Post"


class GraphProcessorWrapper:
    def __init__(self, path='modules/demo/data/9node.json'):
        loader = GraphLoader(path)
        self.g = GraphProcessor(loader.nodes, loader.edges)

    def has_all(self, nodes):
        graph_nodes = [x.name for x in self.g.nodes]
        for n in nodes:
            if n not in graph_nodes:
                return False

        return True

    def find_closest(self, v):
        vehicles = []
        loc = v["currlocation"]["currentPosition"]

        loc = loc.split(',')

        vehicles.append(
            {"id": v["UUID"],
             "latitude": float(loc[1]),
             "longitude": float(loc[0])})
        return self.g.map_vehicles(vehicles)[0]

    def map_vehicles(self, data):
        return [self.g.node_from_id(v["currlocation"]["locationId"]) for v in data]

    def get_graph(self):
        return self.g.nodes, self.g.edges, self.g.incident_matrix

    def node_names(self):
        return [v.name for v in self.g.nodes]


class VrpProcessor:
    def __init__(self, graph_processor):
        self.vrp = VRP()
        self.graphProcessor = graph_processor

    def process(self, metadata):
        global brokenVehicle
        if len(metadata) == 0:
            return []
        nodes, edges, graph = self.graphProcessor.get_graph()

        post_mapping = self.graphProcessor.map_vehicles(metadata)
        for i in range(len(post_mapping)):
            if post_mapping[i] is None:
                print('missing')
                # post_mapping[i] = self.graphProcessor.find_closest(metadata[i])
        print("Got broken vehicle: {}".format(brokenVehicle))
        dropoff = [0] * len(nodes)
        non_broken_vehicles = []
        for i, v in enumerate(metadata):
            for loc in v['dropOffLocations']:
                print(loc['locationId'])
                target = self.graphProcessor.g.node_from_id(loc['locationId'])

                if target is None:  # skip deliveries to nodes not in my graph
                    continue

                idx = nodes.index(target)
                dropoff[idx] += loc['dropoffVolumeM3']
            if v['UUID'] != brokenVehicle:
                non_broken_vehicles.append(v)
            else:
                del post_mapping[i]

        start = [nodes.index(x) for x in post_mapping]
        # extract capacities from individual vehicles
        capacity = [int(x["metadata"]["LoadCapacity"]) for x in non_broken_vehicles]

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
                "info": "This parcel must be delivered to location " + str(node.id),
                "position": "{},{}".format(node.lon, node.lat)
            })

        return converted_route

    def has_all_nodes_from(self, vehicles):
        nodes = []
        for clo in vehicles:
            for node in clo['dropOffLocations']:
                nodes.append(node['locationId'])
        return self.graphProcessor.has_all(nodes)

    def get_node_names(self):
        return self.graphProcessor.node_names()


xborderVrpSouthZagreb = VrpProcessor(graph_processor=GraphProcessorWrapper('modules/demo/data/zagreb_south.json'))
zagrebSVrP = VrpProcessor(graph_processor=GraphProcessorWrapper('modules/demo/data/zagreb_south.json'))
zagrebNVrp = VrpProcessor(graph_processor=GraphProcessorWrapper('modules/demo/data/zagreb_north.json'))
xborderVrpNorthZagreb = VrpProcessor(graph_processor=GraphProcessorWrapper('modules/demo/data/zagreb_north.json'))

xborderVrpSouthAthens = VrpProcessor(graph_processor=GraphProcessorWrapper('modules/demo/data/atene_south.json'))
athensSVrP = VrpProcessor(graph_processor=GraphProcessorWrapper('modules/demo/data/atene_south.json'))
athensNVrp = VrpProcessor(graph_processor=GraphProcessorWrapper('modules/demo/data/atene_north.json'))
xborderVrpNorthAthens = VrpProcessor(graph_processor=GraphProcessorWrapper('modules/demo/data/atene_north.json'))
generalAthens = VrpProcessor(graph_processor=GraphProcessorWrapper('modules/demo/data/atene.json'))


class Event(Resource):

    def get(self):
        return jsonify({"success": True, "message": "Please use POST request"})

    def post(self):
        global brokenVehicle, parcelData
        json = request.get_json(force=True)
        event = json['event']

        print(json)

        print("Posting to SIOT 1")
        # data = requests.post(SIOT_URL + "newEvent", json=json)
        # print(data.content)
        if "broken" in event["type"].lower():
            vehicle = json['vehicle']
            brokenVehicle = vehicle['UUID']
        elif "parcel" in event['type'].lower():
            parcelData = json

        return jsonify({
            "success": True,
            "message": "Processing event for vehicle {}".format(vehicle['UUID'])
        })


class RecReq(Resource):

    def get(self):
        return jsonify({"success": True, "message": "Please use POST request"})

    def msb_forward(self, payload):

        timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S.000Z")
        data = {"recommendations": []}
        counter = 1
        for clo in payload:
            route_plan = {
                "clo": clo["UUID"],
                "plan": {
                    "uuid": "{}-plan-{}-{}".format(org_by_name(clo["UUID"]).replace(' ', ""), timestamp, counter),
                    "organization": org_by_name(clo["UUID"]),
                    "execution_date": timestamp,
                    "steps": []
                }
            }
            counter += 1

            for step in clo["route"]:
                location_split = step["position"].split(',')

                transformed_step = {
                    "location": {
                        "lat": float(location_split[1]),
                        "lng": float(location_split[0])
                    },
                    "station": uuid_by_name(step["locationId"]),
                    "station_type": "post",
                    "load": [str(uuid.uuid4()) for _ in range(randint(0, 4))],
                    "unload": [str(uuid.uuid4()) for _ in range(randint(0, 4))]
                }
                route_plan["plan"]["steps"].append(transformed_step)

            data["recommendations"].append(route_plan)

        print(data)
        response = requests.post(MSB_FWD, json=data)
        print("Posted data to", MSB_FWD)
        print(response.content)

    def post(self):
        data = request.get_json(force=True)
        global brokenVehicle
        print(data)

        vehicle_metadata = data['clos']
        filtered = []  # ignore posts
        for clo in vehicle_metadata:
            if 'LoadCapacity' in clo['metadata']:
                filtered.append(clo)

        vehicle_metadata = filtered

        routes = []
        evt_type = data["event"]["event_type"]
        if "info" in data["event"] and "athens" in data["event"]["info"].lower():
            routes = self.process_athens(data, evt_type, routes, vehicle_metadata)
        else:
            routes = self.process_zagreb(data, evt_type, routes, vehicle_metadata)
        if len(vehicle_metadata) < 1:
            return jsonify({"msg": "No vehicles"})

        try:
            self.msb_forward(routes)
        except Exception as e:
            print("Something went wrong at forwarding to MSB", e)

        return jsonify({"clos": routes})

    def process_zagreb(self, data, evt_type, routes, vehicle_metadata):
        global brokenVehicle
        if "event" in data:
            if "broken" in evt_type:
                brokenVehicle = data["event"]["initiator"]
                print("Broken event initiator:", brokenVehicle)
                print("Processing VRP data.")

                if zagrebSVrP.has_all_nodes_from(vehicle_metadata):
                    routes = zagrebSVrP.process(vehicle_metadata)
                else:
                    routes = zagrebNVrp.process(vehicle_metadata)
            elif "parcel" in evt_type:
                print("Processing VRP data.")
                if zagrebSVrP.has_all_nodes_from(vehicle_metadata):
                    routes = zagrebSVrP.process(vehicle_metadata)
                else:
                    routes = zagrebNVrp.process(vehicle_metadata)
            elif "CrossBorder" in evt_type:
                xvehicles_south = []
                xvehicles_north = []
                for v in vehicle_metadata:
                    if v["currlocation"]["locationId"] in xborderVrpNorthZagreb.get_node_names():
                        xvehicles_north.append(v)
                    else:
                        xvehicles_south.append(v)
                print("Running VRP 1")
                routes1 = xborderVrpSouthZagreb.process(xvehicles_south)
                print("Running VRP 2")
                routes2 = xborderVrpNorthZagreb.process(xvehicles_north)
                routes = routes1 + routes2
        return routes

    def process_athens(self, data, evt_type, routes, vehicle_metadata):
        global brokenVehicle
        if "event" in data:
            if "broken" in evt_type:
                brokenVehicle = data["event"]["initiator"]
                print("Broken event initiator:", brokenVehicle)
                print("Processing VRP data.")

                routes = generalAthens.process(vehicle_metadata)
            elif "request" in evt_type.lower():
                print("Processing VRP data.")
                routes = generalAthens.process(vehicle_metadata)
            elif "CrossBorder" in evt_type:
                xvehicles_south = []
                xvehicles_north = []
                for v in vehicle_metadata:
                    if v["currlocation"]["locationId"] in xborderVrpNorthZagreb.get_node_names():
                        xvehicles_north.append(v)
                    else:
                        xvehicles_south.append(v)
                print("Running VRP 1")
                routes1 = xborderVrpSouthAthens.process(xvehicles_south)
                print("Running VRP 2")
                routes2 = xborderVrpNorthAthens.process(xvehicles_north)
                routes = routes1 + routes2
        return routes


class CognitiveAdvisorAPI:
    def __init__(self, port=5000, host='0.0.0.0'):
        self._port = port
        self._host = host
        self._app = Flask(__name__)
        self._api = Api(self._app)
        self._add_endpoints()

    def _add_endpoints(self):
        self._register_endpoint('/api/adhoc/recommendationRequest', RecReq)
        self._register_endpoint('/api/adhoc/newEvent', Event)

    def _register_endpoint(self, endpoint_name, class_ref):
        self._api.add_resource(class_ref, endpoint_name)

    def start(self):
        serve(self._app, host=self._host, port=self._port)


if __name__ == '__main__':
    server = CognitiveAdvisorAPI()
    server.start()
