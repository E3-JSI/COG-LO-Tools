import os

from flask import Flask, request
from flask_jsonpify import jsonify
from flask_restful import Resource
from waitress import serve

from ..create_graph.config.config_parser import ConfigParser
from ..create_graph.create_graph import JsonGraphCreator
from ..cvrp.processor.vrp_processor import VrpProcessor
from ..partitioning.graph_partitioning_preprocess import GraphPreprocessing
from ..utils.clo_update_handler import CloUpdateHandler

app = Flask(__name__)
vrpProcessorReferenceSloCro = None
vrpProcessorReferenceElta = None

config_parser = ConfigParser()

"""
Example POST MSG:
{
	"eventType": "pickupRequest",
	"useCase": "SLO-CRO",
	"CLOS": [{
			"UUID": "UUID-1",
			"currentLocation": "A416",
			"capacity": 200,
			"parcels": [{
				"UUIDParcel": "ABCD",
				"weight": 2,
				"destination": "A89"
			}]
		}
	],
	"orders": [{
			"UUIDParcel": "Parcel UUID1",
			"UUIDRequest": "Request UUID1",
			"weight": 15,
			"destination": "A89",
			"pickup": "A2",
			"timestamp": 123456679
		},
		{
			"UUIDParcel": "Parcel UUID2",
			"UUIDRequest": "Request UUID2",
			"weight": 15,
			"destination": "A239",
			"pickup": "A2",
			"timestamp": 123456679
		}
	]
}
"""


class RecReq(Resource):
    @staticmethod
    def init_vrp(use_case):
        """" Init VRP processor instance based on use-case defined in parameter"""
        graph_processors = GraphPreprocessing.extract_graph_processors(use_case)
        return VrpProcessor(graph_processors, use_case)

    def msb_forward(self, payload, key):
        pass
        # check api_ijs.py for this code

    @staticmethod
    def process_pickup_requests(clos, requests, vrp_processor_ref):
        print("Processing Pickup Delivery Request for ", len(clos), 'vehicles')
        vehicles = vrp_processor_ref.parse_vehicles(clos)
        deliveries = vrp_processor_ref.parse_deliveries(clos, requests)

        return vrp_processor_ref.process(vehicles, deliveries)

    @staticmethod
    def process_broken_clo(clos, broken_clo, vrp_processor_ref):
        print("Processing Broken CLO for ", len(clos), 'vehicles')
        vehicles = vrp_processor_ref.parse_vehicles(clos)
        deliveries = vrp_processor_ref.parse_deliveries(clos, broken_clo["parcels"])

        return vrp_processor_ref.process(vehicles, deliveries)


@app.route("/api/adhoc/getRecommendation", methods=['POST'])
def handle_recommendation_request():
    global vrpProcessorReferenceSloCro
    global vrpProcessorReferenceElta

    """Main entry point for HTTP request"""
    data = request.get_json(force=True)

    if "useCase" not in data or "eventType" not in data:
        return {"message": "Parameter 'eventType' or 'useCase' is missing"}
    evt_type = data["eventType"]
    use_case = data["useCase"]

    if use_case != "SLO-CRO" and use_case != "ELTA":
        return {"message": "Parameter 'useCase' can have value 'SLO-CRO' or 'ELTA'."}

    # Initialize vrpProcessor if not yet initialized
    if use_case == "SLO-CRO" and vrpProcessorReferenceSloCro is None:
        vrpProcessorReferenceSloCro = RecReq.init_vrp(use_case)
    elif use_case == "ELTA" and vrpProcessorReferenceElta is None:
        vrpProcessorReferenceElta = RecReq.init_vrp(use_case)

    if use_case == "SLO-CRO":
        vrp_processor_ref = vrpProcessorReferenceSloCro
    else:
        vrp_processor_ref = vrpProcessorReferenceElta
    if evt_type == "brokenVehicle":
        if "CLOS" not in data or "BrokenVehicle" not in data:
            return {"message": "Parameter 'CLOS' or 'BrokenVehicle' is missing"}
        clos = data["CLOS"]
        broken_clo = data["BrokenVehicle"]
        recommendations = RecReq.process_broken_clo(clos, broken_clo, vrp_processor_ref)
        return jsonify(recommendations)
    elif evt_type == "pickupRequest":
        if "CLOS" not in data or "orders" not in data:
            return {"message": "Parameter 'CLOS' or 'orders' is missing"}
        clos = data["CLOS"]
        requests = data["orders"]
        recommendations = RecReq.process_pickup_requests(clos, requests, vrp_processor_ref)
        return jsonify(recommendations)
    elif evt_type == "crossBorder":
        print("cross border event received")
    else:
        return jsonify({"message": "Invalid event type: {}".format(evt_type)})


@app.route("/api/clo/newCLOs", methods=['POST'])
def new_clos():
    """
    API route used for handling request for newCLOs.
    This method checks if graph needs to be rebuilt or updated.
    """
    global vrpProcessorReferenceElta
    global vrpProcessorReferenceSloCro

    """Main entry point for HTTP request"""
    data = request.get_json(force=True)

    if "CLOS" not in data or "useCase" not in data:
        return {"message": "Parameter 'CLOS' or 'useCase' is missing"}
    clos = data["CLOS"]  # Extract array of CLOs
    use_case = data["useCase"]
    csv_file = config_parser.get_csv_path(use_case)

    if use_case != "SLO-CRO" and use_case != "ELTA":
        return {"message": "Parameter 'useCase' can have value 'SLO-CRO' or 'ELTA'."}

    needs_rebuild = CloUpdateHandler.handle_new_clo_request(clos, csv_file)
    if needs_rebuild or not os.path.exists(config_parser.get_graph_path(use_case)):
        creator = JsonGraphCreator()
        creator.create_json_graph(use_case)

        # Delete existing pickle file for graph
        pickle_path = config_parser.get_pickle_path(use_case)
        if os.path.exists(pickle_path):
            os.remove(pickle_path)

        # Remove use case specific
        if use_case == "SLO-CRO":
            slo_path = config_parser.get_slo_graph_path()
            slo_pickle_path = config_parser.get_slo_pickle_path()
            cro_path = config_parser.get_cro_graph_path()
            cro_pickle_path = config_parser.get_cro_pickle_path()
            if os.path.exists(slo_path):
                os.remove(slo_path)
            if os.path.exists(slo_pickle_path):
                os.remove(slo_pickle_path)
            if os.path.exists(cro_path):
                os.remove(cro_path)
            if os.path.exists(cro_pickle_path):
                os.remove(cro_pickle_path)
            vrpProcessorReferenceSloCro = None
        else:
            vrpProcessorReferenceElta = None

    return {"success": True}


class CognitiveAdvisorAPI:
    def __init__(self, port=5000, host="0.0.0.0"):
        self._port = port
        self._host = host

    def start(self):
        serve(app, host=self._host, port=self._port)
