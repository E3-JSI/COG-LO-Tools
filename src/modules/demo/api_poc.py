import csv
import os
import json

import requests
from flask import Flask, request
from flask_jsonpify import jsonify
from flask_restful import Resource
from waitress import serve

from ..create_graph.config.config_parser import ConfigParser
from ..create_graph.create_graph import JsonGraphCreator
from ..create_graph.methods import methods
from ..cvrp.processor.vrp_processor import VrpProcessor
from ..partitioning.graph_partitioning_preprocess import GraphPreprocessing
from ..utils.clo_update_handler import CloUpdateHandler
from ..utils.input_output import InputOutputTransformer
from ..utils.tsp import Tsp

app = Flask(__name__)
vrpProcessorReferenceSloCro = None
vrpProcessorReferenceElta = None

config_parser = ConfigParser()
msb_post_url = "https://msb.cog-lo.eu/api/publish"
response_validation_url = "http://db.cog-lo.eu/postRecommendation"
recommendation_post_verification_url = "http://116.203.13.198/api/postRecommendation"
planner_tsp_api_url = "localhost:1234/api/tsp"

# Generic message expected to be returned after request to CA arrived
generic_message_received_response = {
    "status": 1,
    "msg": "The CA has received a new request for optimization"
}


def process_new_CLOs_request(data):
    global vrpProcessorReferenceElta
    global vrpProcessorReferenceSloCro

    if "clos" not in data or "useCase" not in data:
        return {"message": "Parameter 'clos' or 'useCase' is missing"}
    clos = data["clos"]  # Extract array of CLOs
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
            vrpProcessorReferenceSloCro = None
        else:
            vrpProcessorReferenceElta = None

    return {"success": True}


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
    def process_pickup_requests(evt_type, clos, requests, vrp_processor_ref, use_case):
        print("Processing Pickup Delivery Request for ", len(clos), 'vehicles')
        vehicles = vrp_processor_ref.parse_vehicles(clos)
        deliveries = vrp_processor_ref.parse_deliveries(evt_type, clos, requests, use_case)

        return vrp_processor_ref.process(vehicles, deliveries, evt_type, use_case)

    @staticmethod
    def process_broken_clo(evt_type, clos, broken_clo, vrp_processor_ref, use_case):
        print("Processing Broken CLO for ", len(clos), 'vehicles')
        vehicles = vrp_processor_ref.parse_vehicles(clos)
        deliveries = vrp_processor_ref.parse_deliveries(evt_type, clos, broken_clo, use_case)

        return vrp_processor_ref.process(vehicles, deliveries, evt_type, use_case)

    @staticmethod
    def process_cross_border_request(evt_type, clos, requests, vrp_processor_ref, use_case):
        print("Processing Cross Border Delivery Request for ", len(clos), 'vehicles')
        vehicles = vrp_processor_ref.parse_vehicles(clos)
        deliveries = vrp_processor_ref.parse_deliveries(evt_type, clos, requests, use_case)

        return vrp_processor_ref.process(vehicles, deliveries, evt_type, use_case)

    @staticmethod
    def post_response_msb(UUID, recommendations):
        # default=str set because we want to serialize Date objects as well
        json_for_serialization = {
            "recommendation": recommendations
        }
        headers = {'Content-type': 'application/json'}
        content = {
            "request": UUID,
            "topic": "recommendations",
            "sender": "COGLOmoduleCA",
            "message": json_for_serialization
        }

        with open('response.json', 'w') as outfile:
            json.dump(content, outfile)

        try:
            response = requests.post(response_validation_url, json = recommendations, headers=headers, verify=False).json()
            print("validation code for recommendations message: ", response)
        except Exception as ex:
            print("Error occurred while validating response in validation service", ex)

        try:
            response = requests.post(msb_post_url, json = content, headers=headers, verify=False)
            print("response from MSB:", response)
        except Exception as ex:
            print("Error occurred while posting response to MSB", ex)

@app.route("/api/adhoc/getRecommendation", methods=['POST'])
def handle_recommendation_request():
    # TODO: Make generic_message_received_response synchronous and all other operations asynchronous
    """

    // =====================================================
    // REQUEST
    // =====================================================

    Message received will always be in the following format. This info was given by company IntraSoft,
    integration work-package leader on 5.6.2020. clos = Vehicles, parcels = orders.
    {
        organization: "ELTA"
        clos: http://app.cog-lo.eu/getCLOs?org=ELTA, # Vehicles
        parcels: http://app.cog-lo.eu/getParcels?org=ELTA, #Orders
        event: null (for the daily plan),
        request: "request UUID"
    }

    Example of "clos" field:
    {
      "status": 1,
      "msg": "18 CLOs found",
      "clos": [
        {
          "id": "ELTApostoffice12188",
          "info": {
            "id": "ELTApostoffice12188",
            "name": "ELTA post office A7",
            "type": "station",
            "subtype": "post office",
            "capacity": null,
            "location": {
              "city": null,
              "country": null,
              "station": null,
              "latitude": 38.0018005371,
              "longitude": 23.7500495911,
              "postal_code": null
            },
            "description": null,
            "physical_id": "12188",
            "capacity_unit": null,
            "working_hours": null
          },
          "state": null
        },
        {
          "id": "ELTAvehicle001",
          "info": {
            "id": "ELTAvehicle001",
            "name": "ELTA vehicle 001",
            "type": "transportation means",
            "subtype": "vehicle",
            "capacity": null,
            "location": {
              "city": null,
              "country": null,
              "station": null,
              "latitude": null,
              "longitude": null,
              "postal_code": null
            },
            "description": "truck",
            "physical_id": "001",
            "capacity_unit": null,
            "working_hours": null
          },
          "state": {
            "plan": "ELTAplan39",
            "driver": null,
            "status": null,
            "parcels": null,
            "location": {
              "city": null,
              "country": null,
              "station": null,
              "latitude": 37.9870396023,
              "longitude": 23.7520077519,
              "postal_code": null
            },
            "load_volume": null,
            "capacity_unit": null,
            "delay_minutes": null,
            "completed_plan": {
              "steps": [
                {
                  "id": 1,
                  "rank": 1,
                  "unload": [
                    "e5ece9c1-c2bf-443a-881a-04c08bbd1256"
                  ],
                  "complete": 1,
                  "due_time": null,
                  "location": {
                    "city": null,
                    "country": null,
                    "station": "PSpostoffice3701",
                    "latitude": 45.755153656,
                    "longitude": 15.9372501373,
                    "postal_code": null
                  },
                  "dependency": {
                    "plan": null,
                    "plan_step": null
                  }
                }
              ]
            },
            "remaining_plan": {
              "steps": [
                {
                  "id": 2,
                  "load": [
                    "5ae413b4-530d-437f-bd1b-7ef1994c0fca",
                    "694ef5bb-6a06-42e2-a589-6881dedc78ff"
                  ],
                  "rank": 2,
                  "unload": [
                    "d3ef0800-931c-46be-a458-c509465df1c3",
                    "1d1b75d4-fa23-423a-af55-4c54c2db8136",
                    "90df51cd-c06a-4918-90ae-8cdefdb57bd4"
                  ],
                  "complete": 0,
                  "due_time": null,
                  "location": {
                    "city": null,
                    "country": null,
                    "station": "PSpostoffice5016",
                    "latitude": 45.7793617249,
                    "longitude": 15.9955749512,
                    "postal_code": null
                  },
                  "dependency": {
                    "plan": null,
                    "plan_step": null
                  }
                },
                {
                  "id": 3,
                  "load": [
                    "8913e033-12d8-4011-a986-ed880fc94dd3",
                    "a4ccbe6a-4bb4-4c94-85a5-80ff4bce45cb"
                  ],
                  "rank": 3,
                  "unload": [
                    "ccac88e7-a605-4d3b-b920-8e93be7abcad",
                    "22f4b9d0-204e-46f2-8252-eb3698dc05e3"
                  ],
                  "complete": 0,
                  "due_time": null,
                  "location": {
                    "city": null,
                    "country": null,
                    "station": "HPpostoffice7674",
                    "latitude": 45.7909164429,
                    "longitude": 15.9506902695,
                    "postal_code": null
                  },
                  "dependency": {
                    "plan": null,
                    "plan_step": null
                  }
                },
                {
                  "id": 4,
                  "load": [
                    "ccbae818-b27d-451b-8861-765dd241c59c",
                    "960ecda8-0290-42c4-b230-41f3268e8d56"
                  ],
                  "rank": 4,
                  "complete": 0,
                  "due_time": null,
                  "location": {
                    "city": null,
                    "country": null,
                    "station": "HPpostoffice639",
                    "latitude": 45.8179473877,
                    "longitude": 15.9740610123,
                    "postal_code": null
                  },
                  "dependency": {
                    "plan": null,
                    "plan_step": null
                  }
                },
                {
                  "id": 5,
                  "load": [
                    "037f92e3-cef3-4360-ad42-3c808d398f2f",
                    "02e43622-2919-4041-a522-ac355e5ce93e",
                    "2334871c-c5c2-40c2-891e-fa2a48ea7167",
                    "cfe75035-a55a-423a-a2a3-fcacdd07729a"
                  ],
                  "rank": 5,
                  "complete": 0,
                  "due_time": null,
                  "location": {
                    "city": null,
                    "country": null,
                    "station": "HPpostoffice3148",
                    "latitude": 45.8144264221,
                    "longitude": 16.019744873,
                    "postal_code": null
                  },
                  "dependency": {
                    "plan": null,
                    "plan_step": null
                  }
                }
              ]
            },
            "available_space": null,
            "availability_prediction": {
              "time": null,
              "location": {
                "city": null,
                "country": null,
                "station": null,
                "latitude": null,
                "longitude": null,
                "postal_code": null
              }
            }
          }
        }
      ]
    }

    Example of 'parcels' fields:
    {
      "status": 1,
      "msg": "8 parcels found",
      "parcels": [
        {
          "id": "EKOLorder346925744",
          "state": "unserved",
          "issued": null,
          "source": {
            "city": null,
            "country": null,
            "station": "EKOLstation1",
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "deadline": null,
          "payweight": 13700,
          "destination": {
            "city": null,
            "country": null,
            "station": null,
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "organization": "EKOL",
          "service_type": null,
          "customs_checkpoint": null
        },
        {
          "id": "EKOLorder366874833",
          "state": "unserved",
          "issued": null,
          "source": {
            "city": null,
            "country": null,
            "station": "EKOLstation1",
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "deadline": null,
          "payweight": 10400,
          "destination": {
            "city": null,
            "country": null,
            "station": null,
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "organization": "EKOL",
          "service_type": null,
          "customs_checkpoint": null
        },
        {
          "id": "EKOLorder366906596",
          "state": "unserved",
          "issued": null,
          "source": {
            "city": null,
            "country": null,
            "station": "EKOLstation1",
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "deadline": null,
          "payweight": 23800,
          "destination": {
            "city": null,
            "country": null,
            "station": null,
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "organization": "EKOL",
          "service_type": null,
          "customs_checkpoint": null
        },
        {
          "id": "EKOLorder366909632",
          "state": "unserved",
          "issued": null,
          "source": {
            "city": null,
            "country": null,
            "station": "EKOLstation1",
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "deadline": null,
          "payweight": 10400,
          "destination": {
            "city": null,
            "country": null,
            "station": null,
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "organization": "EKOL",
          "service_type": null,
          "customs_checkpoint": null
        },
        {
          "id": "EKOLorder366926724",
          "state": "unserved",
          "issued": null,
          "source": {
            "city": null,
            "country": null,
            "station": "EKOLstation1",
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "deadline": null,
          "payweight": 8000,
          "destination": {
            "city": null,
            "country": null,
            "station": null,
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "organization": "EKOL",
          "service_type": null,
          "customs_checkpoint": null
        },
        {
          "id": "EKOLorder366929776",
          "state": "unserved",
          "issued": null,
          "source": {
            "city": null,
            "country": null,
            "station": "EKOLstation1",
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "deadline": null,
          "payweight": 3200,
          "destination": {
            "city": null,
            "country": null,
            "station": null,
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "organization": "EKOL",
          "service_type": null,
          "customs_checkpoint": null
        },
        {
          "id": "EKOLorder366933058",
          "state": "unserved",
          "issued": null,
          "source": {
            "city": null,
            "country": null,
            "station": "EKOLstation1",
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "deadline": null,
          "payweight": 1000,
          "destination": {
            "city": null,
            "country": null,
            "station": null,
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "organization": "EKOL",
          "service_type": null,
          "customs_checkpoint": null
        },
        {
          "id": "EKOLorder366933280",
          "state": "unserved",
          "issued": null,
          "source": {
            "city": null,
            "country": null,
            "station": "EKOLstation1",
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "deadline": null,
          "payweight": 1400,
          "destination": {
            "city": null,
            "country": null,
            "station": null,
            "latitude": null,
            "longitude": null,
            "postal_code": null
          },
          "organization": "EKOL",
          "service_type": null,
          "customs_checkpoint": null
        }
      ]
    }

    // =====================================================
    // RESPONSE
    // =====================================================

    Response sent synchronously back after receiving the request is in the following format. Defined on 5.6.2020.
    {
        "status": 1,
        "msg": "The CA has received a new request for optimization"
    }

    Response given after calculating recommendations should be send to MSB and is in the following format.
    Defined on 5.6.2020.
    {
    topic: "Recommendation",
    sender: "COGLOmoduleCA",
    message:
        {
            request: "request UUID"
            recommendation: http://app.cog-lo.eu/getRecommendation?id=EKOLrecommendation1
        }
    }

    Example of recommendation message from 5.6.2020:
    {
        "status": 1,
        "msg": "20 plans contained in the recommendation",
        "recommendation": {
            "organization": "EKOL",
            "id": "EKOLrecommendation1",
            "cloplans": [
                {
                    "clo": "EKOLvehicle34DL8536",
                    "plan": {
                        "organization": "EKOL",
                        "execution_date": "2020-01-13 17:25:35.000000",
                        "recommendation": "EKOLrecommendation1",
                        "id": "EKOLplan6",
                        "steps": [
                            {
                                "id": 1,
                                "load": [
                                    "EKOLcontainer34G80597"
                                ],
                                "rank": 1,
                                "complete": 1,
                                "due_time": null,
                                "location": {
                                "city": null,
                                "country": null,
                                "station": "EKOLcontainer34G80597",
                                "latitude": 47.4481658936,
                                "longitude": 19.0626106262,
                                "postal_code": null
                                },
                                "dependency": {
                                    "plan": null,
                                    "plan_step": null
                                }
                            }
                        ]
                    }
                },
                {
                    "clo": "EKOLvehicleTruck1",
                    "plan": {
                        "organization": "EKOL",
                        "execution_date": "2020-01-13 17:25:33.000000",
                        "recommendation": "EKOLrecommendation1",
                        "id": "EKOLplan4",
                        "steps": [
                            {
                                "id": 1,
                                "load": [
                                    "EKOLcontainerTrailer1"
                                ],
                                "rank": 1,
                                "complete": 1,
                                "due_time": null,
                                "location": {
                                "city": null,
                                "country": null,
                                "station": "EKOLwarehouse2",
                                "latitude": 50.25258255,
                                "longitude": 19.1939735413,
                                "postal_code": null
                                },
                                "dependency": {
                                    "plan": null,
                                    "plan_step": null
                                }
                            },
                            {
                                "id": 2,
                                "pack": [
                                    {
                                        "content": [
                                            "EKOLorder366929776"
                                        ],
                                        "container": "EKOLcontainerTrailer1"
                                    }
                                ],
                                "rank": 2,
                                "complete": 0,
                                "due_time": null,
                                "location": {
                                "city": null,
                                "country": null,
                                "station": "EKOLwarehouse2",
                                "latitude": 50.25258255,
                                "longitude": 19.1939735413,
                                "postal_code": null
                                },
                                "dependency": {
                                    "plan": null,
                                    "plan_step": null
                                }
                            }
                         ]
                    }
                }
            ]
        }
    }
    """

    global vrpProcessorReferenceSloCro
    global vrpProcessorReferenceElta

    """Main entry point for HTTP request"""
    received_request = request.get_json(force=True)

    print("received getRecommendation request", received_request)

    # transforms received message for internal structures
    try:
        data = InputOutputTransformer.parse_received_recommendation_message(received_request)
    except ValueError as error:
        return str(error)

    # needed for response handling
    request_id = received_request["request"]
    use_case = data['useCase']

    ##Errors
    if use_case != "SLO-CRO" and use_case != "ELTA":
        return {"message": "Parameter 'useCase' can have value 'SLO-CRO' or 'ELTA'."}
    if "useCase" not in data or "eventType" not in data:
        return {"message": "Parameter 'eventType' or 'useCase' is missing"}
    evt_type = data["eventType"]
    use_case = data["useCase"]

    ##Use Case SLO-CRO
    if use_case == "SLO-CRO":
        print("processing SLO-CRO use-case request")
        if vrpProcessorReferenceSloCro is None:     #initialize VRP
            vrpProcessorReferenceSloCro = RecReq.init_vrp(use_case)
        vrp_processor_ref = vrpProcessorReferenceSloCro

        # Extract 'clos' which should be a field in each request
        if "clos" not in data:
            return {"msg": "Parameter 'clos' is missing", "status": 0}
        clos = data["clos"]

        if evt_type == "brokenVehicle":
            if "clos" not in data or "brokenVehicle" not in data:
                return {"msg": "Parameter 'clos' or 'BrokenVehicle' is missing", "status": 0}
            broken_clo = data["brokenVehicle"]
            recommendations = RecReq.process_broken_clo(evt_type, clos, broken_clo, vrp_processor_ref, use_case)
        elif evt_type == "pickupRequest":
            if "clos" not in data or "orders" not in data:
                return {"msg": "Parameter 'clos' or 'orders' is missing", "status": 0}
            requests = data["orders"]
            recommendations = RecReq.process_pickup_requests(evt_type, clos, requests, vrp_processor_ref, use_case)
        elif evt_type == "crossBorder":
            print("cross border event received")
            if "clos" not in data:
                return {"msg": "Parameter 'clos' is missing", "status": 0}
            requests = []
            for clo in clos:
                parcels = clo["parcels"]
                for parcel in parcels:
                    parcel["currentLocation"] = clo["currentLocation"]
                    requests.append(parcel)
            recommendations = RecReq.process_cross_border_request(evt_type, clos, requests, vrp_processor_ref, use_case)
        else:
            return jsonify({"message": "Invalid event type: {}".format(evt_type), "status": 0})

        # Executes TSP algorithm upon calculated recommendations by our VRP
        recommendations = Tsp.order_recommendations(recommendations)

        # Prepare output message from calculated recommendations
        response = InputOutputTransformer.prepare_output_message(recommendations, use_case, request_id)

        # This piece of code posts optimization response to MSB
        RecReq.post_response_msb(request_id, response)

        # Always return generic message stating that request was received and is due to be processed
        return generic_message_received_response

    ##Use Case ELTA
    elif use_case == "ELTA":
        print("processing ELTA usecase")
        ### VRP INICIALIZATION AND MESSAGE PREPROCESSING
        transform_map_dict = methods.get_orders_coordinates(data)
        if evt_type is None:
            data_request, data_CLOs = methods.proccess_elta_event(evt_type, data)
            res = process_new_CLOs_request(data_CLOs)  # make graph build
        else:
            data_request = methods.proccess_elta_event(evt_type, data)

        if vrpProcessorReferenceElta is None:     #initialize VRP
            vrpProcessorReferenceElta = RecReq.init_vrp(use_case)
        vrp_processor_ref = vrpProcessorReferenceElta

        if "clos" not in data_request:
            return {"msg": "Parameter 'clos' is missing", "status": 0}
        clos = data_request["clos"]

        ### MESSAGE PROCESSING ....
        if evt_type is None:
            if "orders" not in data_request:
                return {"msg": "Parameter 'orders' is missing", "status": 0}
            data_requests = data_request["orders"]
            recommendations = RecReq.process_pickup_requests(evt_type, clos, data_requests, vrp_processor_ref, use_case)
        elif evt_type == "brokenVehicle":
            if "brokenVehicle" not in data_request:
                return {"msg": "Parameter 'BrokenVehicle' is missing", "status": 0}
            broken_clo = data_request["brokenVehicle"]
            recommendations = RecReq.process_broken_clo(evt_type, clos, broken_clo, vrp_processor_ref, use_case)

        elif evt_type == "pickupRequest":
            if "orders" not in data_request:
                return {"msg": "Parameter 'orders' is missing", "status": 0}
            data_requests = data_request["orders"]
            recommendations = RecReq.process_pickup_requests(evt_type, clos, data_requests, vrp_processor_ref, use_case)
        else:
            return jsonify({"msg": "Invalid event type: {}".format(evt_type), "status": 0})

        # Executes TSP on given recommendations to order route plan correctly
        recommendations = Tsp.order_recommendations(recommendations)

        # Maps recommendations based on transform_map_dict
        recommendations_mapped = methods.map_coordinates_to_response(recommendations, transform_map_dict)

        # Transforms response in 'JSI' format to the one used for MSB
        response1 = InputOutputTransformer.prepare_output_message(recommendations_mapped, use_case, request_id)

        # restructures steps plan and and lists all the parcels from clusters as a list of locations
        response = methods.order_parcels_on_route(response1)

        # This piece of code posts optimization response to MSB
        RecReq.post_response_msb(request_id, response)

        # Response is always a generic one which just states that CA received request and will process it.
        return generic_message_received_response

@app.route("/api/clo/newCLOs", methods=['POST'])
def new_clos():
    """
    API route used for handling request for newCLOs.
    This method checks if graph needs to be rebuilt or updated.

    Example request:
    {
      "clos": [
        {
          "id": "PSpostofficeS1",
          "info": {
            "name": "Kapele",
            "type": "station",
            "subtype": "post office",
            "organization": "PS",
            "physical_id": "S1",
            "location": {
              "latitude": 45.93107932,
              "longitude": 15.67793203,
              "country": "SLO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "PSpostofficeS2",
          "info": {
            "name": "Globoko",
            "type": "station",
            "subtype": "post office",
            "organization": "PS",
            "physical_id": "S2",
            "location": {
              "latitude": 45.95503619,
              "longitude": 15.63656969,
              "country": "SLO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "PSpostofficeS3",
          "info": {
            "name": "Jesenice",
            "type": "station",
            "subtype": "post office",
            "organization": "PS",
            "physical_id": "S3",
            "location": {
              "latitude": 45.85945758,
              "longitude": 15.68964069,
              "country": "SLO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "PSpostofficeS4",
          "info": {
            "name": "Topliska cesta",
            "type": "station",
            "subtype": "post office",
            "organization": "PS",
            "physical_id": "S4",
            "location": {
              "latitude": 45.89320611,
              "longitude": 15.6212996,
              "country": "SLO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "PSpostofficeS5",
          "info": {
            "name": "Cerklje ob Krki",
            "type": "station",
            "subtype": "post office",
            "organization": "PS",
            "physical_id": "S5",
            "location": {
              "latitude": 45.88461153,
              "longitude": 15.5231045,
              "country": "SLO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "PSpostofficeS6",
          "info": {
            "name": "Ulica 11. novembra",
            "type": "station",
            "subtype": "post office",
            "organization": "PS",
            "physical_id": "S6",
            "location": {
              "latitude": 45.942039,
              "longitude": 15.478103,
              "country": "SLO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "PSpostofficeS7",
          "info": {
            "name": "Kriska vas",
            "type": "station",
            "subtype": "post office",
            "organization": "PS",
            "physical_id": "S7",
            "location": {
              "latitude": 45.89123853,
              "longitude": 15.57086927,
              "country": "SLO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "PSpostofficeS8",
          "info": {
            "name": "Ulica stare pravde",
            "type": "station",
            "subtype": "post office",
            "organization": "PS",
            "physical_id": "S8",
            "location": {
              "latitude": 45.90488128,
              "longitude": 15.59338439,
              "country": "SLO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "PSpostofficeS9",
          "info": {
            "name": "Ulica bratov Gerjovicev",
            "type": "station",
            "subtype": "post office",
            "organization": "PS",
            "physical_id": "S9",
            "location": {
              "latitude": 45.89504865,
              "longitude": 15.66006978,
              "country": "SLO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "HPpostofficeH1",
          "info": {
            "name": "Ul. ?ure Basari?eka 9",
            "type": "station",
            "subtype": "post office",
            "organization": "HP",
            "physical_id": "H1",
            "location": {
              "latitude": 45.839828,
              "longitude": 15.68643,
              "country": "CRO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "HPpostofficeH2",
          "info": {
            "name": "Ul. Ljudevita Gaja 4",
            "type": "station",
            "subtype": "post office",
            "organization": "HP",
            "physical_id": "H2",
            "location": {
              "latitude": 45.803142,
              "longitude": 15.710135,
              "country": "CRO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "HPpostofficeH3",
          "info": {
            "name": "Gori?ka 1",
            "type": "station",
            "subtype": "post office",
            "organization": "HP",
            "physical_id": "H3",
            "location": {
              "latitude": 45.917165,
              "longitude": 15.730056,
              "country": "CRO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "HPpostofficeH4",
          "info": {
            "name": "Ul. Dr. Franje Tu?mana 38a",
            "type": "station",
            "subtype": "post office",
            "organization": "HP",
            "physical_id": "H4",
            "location": {
              "latitude": 45.808406,
              "longitude": 15.816847,
              "country": "CRO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "HPpostofficeH5",
          "info": {
            "name": "Zagreba?ka ul. 45",
            "type": "station",
            "subtype": "post office",
            "organization": "HP",
            "physical_id": "H5",
            "location": {
              "latitude": 45.887335,
              "longitude": 15.697116,
              "country": "CRO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "HPpostofficeH6",
          "info": {
            "name": "Ul. Drage ?vajcara 5",
            "type": "station",
            "subtype": "post office",
            "organization": "HP",
            "physical_id": "H6",
            "location": {
              "latitude": 45.859075,
              "longitude": 15.80341,
              "country": "CRO"
            }
          },
          "state": {
            "status": 1.0
          }
        },
        {
          "id": "HPpostofficeH7",
          "info": {
            "name": "Zagreba?ka ul. 29",
            "type": "station",
            "subtype": "post office",
            "organization": "HP",
            "physical_id": "H9",
            "location": {
              "latitude": 45.878072,
              "longitude": 15.739208,
              "country": "CRO"
            }
          },
          "state": {
            "status": 1.0
          }
        }
      ],
      "useCase": "SLO-CRO"
    }

    """
    """Main entry point for HTTP request"""
    data = request.get_json(force=True)

    print("newCLOs request arrived with payload:", data)

    clos = data["clos"]  # Extract array of CLOs

    # Check for non empty array
    if len(clos) < 1:
        return {
            "status": -1,
            "msg": "List of 'clos' should have at least one field."
        }
    first_clo = clos[0]["info"]
    use_case = first_clo["organization"]

    # for ELTA only update the .csv list of of static locations
    if use_case == "ELTA":
        csv_file_path = config_parser.get_elta_path()
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for json_obj in clos:
                # Extract CLO ID and it's location object. Location object should have fields 'address',
                # 'latitude' and 'longitude'
                id = json_obj["id"]
                info_obj = json_obj["info"]
                location = info_obj["location"]

                # Write rows
                csv_writer.writerow([location["address"], id, location["latitude"], location["longitude"]])
        csv_file.close()
        return {"success": True}
    # for SLO-CRO create a new graph
    else:
        return process_new_CLOs_request(data)


class CognitiveAdvisorAPI:
    def __init__(self, port=5000, host="0.0.0.0"):
        self._port = port
        self._host = host

    def start(self):
        serve(app, host=self._host, port=self._port)
