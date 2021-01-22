import copy
import csv
from datetime import datetime

import numpy as np

from ..create_graph.config.config_parser import ConfigParser
from ..utils.tsp import Tsp
from ..utils.ConflictNodeOrdering import OrderRelations


config_parser = ConfigParser()

ELTA_USE_CASE = "ELTA"
SLO_CRO_USE_CASE = "SLO-CRO"


class InputOutputTransformer:
    @staticmethod
    def clo(uuid, info, state):
        return {
            "UUID": uuid,
            "info": info,
            "state": state
        }

    @staticmethod
    def clo_info(organization, physical_id, type, subtype, description, location, capacity, capacity_unit,
                 working_hours):
        return {
            "organization": organization,
            "physical_id": physical_id,
            "type": type,
            "subtype": subtype,
            "description": description,
            "location": location,
            "capacity": capacity,
            "capacity_unit": capacity_unit,
            "working_hours": working_hours
        }

    @staticmethod
    def location(lat, lon, address=None, postal_code=None, city=None, country=None):
        return {
            "address": address,
            "postal_code": postal_code,
            "city": city,
            "country": country,
            "lat": lat,
            "lon": lon
        }

    @staticmethod
    def parking(uuid, uuids):
        return {
            "container": uuid,
            "content": uuids
        }

    # Vehicle route
    @staticmethod
    def plan(uuid, organization, steps, execution_date=None):
        return {
            "id": uuid,
            "organization": organization,
            "execution_date": execution_date,
            "steps": steps
        }

    @staticmethod
    def recommendation(uuid, plan):
        return {
            "clo": uuid,
            "plan": plan
        }

    @staticmethod
    def parking(uuid, content):
        return {
            "container": uuid,
            "content": content
        }

    @staticmethod
    def itinerary(station, departure, arrival, cutoff):
        return {
            "station": station,
            "departure_time": departure,
            "arrival_time": arrival,
            "cutoff_time": cutoff
        }

    @staticmethod
    def schedule(weekday, start, end):
        return {
            "weekday": weekday,
            "start_time": start,
            "end_time": end
        }

    @staticmethod
    def postalbox_state(location, status, load_size, load_unit):
        return {
            "location": location,
            "status": status,
            "load_size": load_size,
            "load_unit": load_unit
        }

    @staticmethod
    def transportation_state(location, status, prediction, load_volume, available, capacity_unit, driver, plan,
                             completed, remaining, delay):
        return {
            "location": location,
            "status": status,
            "availability_prediction": prediction,
            "load_volume": load_volume,
            "available_space": available,
            "capacity_unit": capacity_unit,
            "driver": driver,
            "plan": plan,
            "completed_plan": completed,
            "remaining_plan": remaining,
            "delay": delay
        }

    @staticmethod
    def parcel(organization, id, source_id, source_type, source_location, destination_id, destination_type,
               destination_location, service_type, payweight, issued, customs):
        return {
            "organization": organization,
            "id": id,
            "source_id": source_id,
            "source_type": source_type,
            "source_location": source_location,
            "destination_id": destination_id,
            "destination_type": destination_type,
            "destination_location": destination_location,
            "service_type": service_type,
            "payweight": payweight,
            "issued": issued,
            "customs_checkpoint": customs
        }

    @staticmethod
    def event(time, event_type, event, initiator, info):
        return {
            "time": time,
            "event_type": event_type,
            "event": event,
            "initiator": initiator,
            "info": info
        }

    @staticmethod
    def packing(container, content):
        return {
            "container": container,
            "content": content
        }

    @staticmethod
    def plan_step(id, rank, order, location, station, station_type, pack, unpack, load, unload, due_time, dependency):
        return {
            "id": id,
            "rank": rank,
            "order": order,
            "location": location,
            "station": station,
            "station_type": station_type,
            "pack": pack,
            "unpack": unpack,
            "load": load,
            "unload": unload,
            "due_time": due_time,
            "dependency": dependency
        }

    @staticmethod
    def get_clo_parcels(clo):
        parcels = []
        if clo["state"]["remaining_plan"] == None or len(clo["state"]["remaining_plan"]["steps"]) == 0:
            return parcels
        else:
            for step in clo["state"]["remaining_plan"]["steps"]:
                parcels.extend(step["unload"])
                print (step["unload"])
                #clo_parcels = copy.deepcopy(parcels)
            return parcels


    @staticmethod
    def parse_received_recommendation_message(json, transformation_map, use_case_graph):
        InputOutputTransformer.validateMessageForValue(json, ["organization"])
        InputOutputTransformer.validateMessageForValue(json, ["request"])
        payload = {
            "useCase": json["organization"],
            "UUIDRequest": json["request"],
        }

        # SLO-CRO use-case and it's parsing remained the same as it was before... we need example message before
        # changing the structure
        if json["organization"] == SLO_CRO_USE_CASE or json["organization"] == "PS" or json["organization"] == "HP":
            payload["useCase"] = SLO_CRO_USE_CASE
            if "event" not in json:
                payload["eventType"] = "pickupRequest"
            else:
                event = json["event"]
                if event is None:  # Procedure for daily plan is the same as for pickupRequest
                    payload["eventType"] = "pickupRequest"
                else:
                    type = event["event_type"]
                    if type == "order":
                        type = "AdHocRequest"
                    elif type == "vehicle":
                        type = "brokenVehicle"
                    elif type == "border":
                        type = "crossBorder"
                    payload["eventType"] = type
        elif json["organization"] == ELTA_USE_CASE:
            payload["useCase"] = ELTA_USE_CASE
            # None is used for dailyPlan
            if "event" not in json:
                payload["eventType"] = None
            else:
                event = json["event"]
                if event is None:
                    payload["eventType"] = None
                else:
                    type = event["event_type"]
                    if type == "order":
                        type = "AdHocRequest"
                    elif type == "vehicle":
                        type = "brokenVehicle"
                    payload["eventType"] = type

        InputOutputTransformer.validateMessageForValue(json, ["clos"])
        clos = json["clos"][:]

        # Array of parcels from the message. This array will help us extracting fields if for example,
        # some CLO only has parcel ID instead of all the information about this parcel!
        parcels_dict = None
        json_parcels = list(json["parcels"])
        if "parcels" in json:
            parcels_dict = {element["id"]: element for element in json_parcels[:]}

        for clo in clos:
            InputOutputTransformer.validateMessageForValue(clo, ["id"])
            clo["UUID"] = clo["id"]

            location = None
            if "state" in clo and "location" in clo["state"]:
                location = clo["state"].pop('location')

                if location["latitude"] is None or location["longitude"] is None:
                    location = None

            InputOutputTransformer.validateMessageForValue(location, ["latitude", "longitude"])
            if payload["useCase"] == SLO_CRO_USE_CASE:
                # Extract station ID from given 'latitude' and 'longitude'.
                # location_station_dict = CloUpdateHandler.extract_location_station_dict(csv_file)
                station_id = InputOutputTransformer.getStationIdOrClosest(
                    location, config_parser.get_csv_path(use_case_graph), transformation_map)
                clo['currentLocation'] = station_id
            else:
                clo['currentLocation'] = [location["latitude"], location["longitude"]]

            if "country" in location and location["country"] is not None:
                clo['country'] = location["country"]
            elif payload["useCase"] == SLO_CRO_USE_CASE:
                clo["country"] = "SLO" if clo["id"].startswith("PS") else "CRO"
            else:
                clo["country"] = "GREECE"

            InputOutputTransformer.validateMessageForValue(clo, ["info"])
            InputOutputTransformer.validateMessageForValue(clo["info"], ["capacity"])
            clo['capacity'] = clo["info"].pop('capacity')

            ########################################################################
            # PARCELS ON THE VEHICLE (CLO)
            ########################################################################

            # Check for state -> remaining_plan -> steps
            if "state" not in clo or clo["state"] is None:
                clo["parcels"] = []

            parcels = InputOutputTransformer.get_clo_parcels(clo)
            clo_parcels = []

            # Has parcels on the vehicle, go through them
            parcel_dict_cpy = copy.deepcopy(parcels_dict)
            for parcel_id in parcels:
                parcel = parcel_dict_cpy[parcel_id]
                InputOutputTransformer.validateMessageForValue(parcel, ["id"])

                parcel['UUIDParcel'] = parcel['id']
                parcel['weight'] = parcel['payweight']

                InputOutputTransformer.validateMessageForValue(parcel, ["destination"])
                destination = parcel['destination']
                parcel["country"] = clo["country"]

                if payload["useCase"] == SLO_CRO_USE_CASE:
                    csv_file = config_parser.get_csv_path(use_case_graph)
                    station_id = InputOutputTransformer.getStationIdOrClosest(destination, csv_file, transformation_map,
                                                                              parcel['id'])
                    parcel['destination'] = station_id
                else:
                    InputOutputTransformer.validateMessageForValue(destination, ["latitude", "longitude"])
                    parcel['destination'] = [destination["latitude"], destination["longitude"]]
                clo_parcels.append(parcel)

            clo["parcels"] = clo_parcels  # Parcels already on the vehicle!
        payload["clos"] = clos

        ########################################################################
        # REMOVE BROKEN VEHICLE FROM CANDIDATES FOR DELIVERIES AND CREATE
        # NEW ORDERS PLAN TO DELIVER REMAINING PARCELS FROM THE BROKEN VEHICLE
        ########################################################################
        print ("test")
        if payload["eventType"] == "brokenVehicle":
            InputOutputTransformer.validateMessageForValue(json["event"], ["info"])
            InputOutputTransformer.validateMessageForValue(json["event"]["info"], ["clo"])
            broken_clo = json["event"]["info"]["clo"]

            # First thing we need to do is remove broken vehicle from the list of "clos" available for deliveries!
            new_clos = []
            for clo in payload["clos"]:
                # This vehicle is no longer useful, because it is broken!
                if clo["id"] == broken_clo["id"]:
                    continue
                new_clos.append(clo)
            payload["clos"] = new_clos

            current_location = None
            InputOutputTransformer.validateMessageForValue(broken_clo, ["info"])
            InputOutputTransformer.validateMessageForValue(broken_clo, ["state"])
            InputOutputTransformer.validateMessageForValue(broken_clo["state"], ["location"])
            InputOutputTransformer.validateMessageForValue(broken_clo["state"]["location"], ["latitude"])
            InputOutputTransformer.validateMessageForValue(broken_clo["state"]["location"], ["longitude"])

            if "state" in broken_clo and "location" in broken_clo["state"]:
                current_location = broken_clo["state"]['location']

                if current_location["latitude"] is None or current_location["longitude"] is None:
                    current_location = None

            if payload["useCase"] == SLO_CRO_USE_CASE:
                station_id = InputOutputTransformer.getStationIdOrClosest(
                    current_location, config_parser.get_csv_path(use_case_graph), transformation_map)
                broken_clo["currentLocation"] = station_id
            else:
                InputOutputTransformer.validateMessageForValue(current_location, ["latitude", "longitude"])
                broken_clo['currentLocation'] = [current_location["latitude"], current_location["longitude"]]

            # Assign location from 'country' field if exists or from name
            broken_vehicle_country = None
            if "country" in current_location:
                broken_vehicle_country = current_location["country"]
            if broken_vehicle_country is None:
                if broken_clo["id"].find("PS") != -1:
                    broken_vehicle_country = "SLO"
                elif broken_clo["id"].find("HP") != -1:
                    broken_vehicle_country = "CRO"
                elif broken_clo["id"].find("ELTA") != -1:
                    broken_vehicle_country = "GREECE"
            broken_clo['country'] = broken_vehicle_country

            InputOutputTransformer.validateMessageForValue(broken_clo, ["state"])

            orders = []  # Zero orders

            # Current vehicle state at breakdown
            vehicle_state = broken_clo["state"]
            InputOutputTransformer.validateMessageForValue(vehicle_state, ["parcels"])
            parcel_ids = InputOutputTransformer.get_clo_parcels(broken_clo)

            # Currently loaded parcels on the vehicle need to be delivered as well!
            for parcel_id in parcel_ids:
                parcel_dict_cpy = copy.deepcopy(parcels_dict)
                if parcel_id not in parcel_dict_cpy:
                    raise ValueError("Parcel with ID " + str(parcel_id) +
                                     " not on the list of 'parcels'!")

                # Package source is the location of the vehicle!
                InputOutputTransformer.validateMessageForValue(vehicle_state, ["location"])

                # Get parcel object.
                vehicle_location = vehicle_state["location"]
                parcel = parcel_dict_cpy[parcel_id]
                parcel['UUIDParcel'] = parcel.pop('id')
                parcel['weight'] = parcel.pop('payweight')
                parcel["country"] = vehicle_location["country"]

                if parcel["country"] is None and payload["useCase"] == SLO_CRO_USE_CASE:
                    parcel["country"] = "SLO" if parcel["UUIDParcel"].startswith("PS") else "CRO"

                # Pickup of these parcels is the vehicle's location!
                if payload["useCase"] == SLO_CRO_USE_CASE:
                    station_id = InputOutputTransformer.getStationIdOrClosest(
                        vehicle_location, config_parser.get_csv_path(use_case_graph), transformation_map)
                    parcel['pickup'] = station_id
                else: #ELTA use case
                    InputOutputTransformer.validateMessageForValue(vehicle_location, ["latitude", "longitude"])
                    parcel['pickup'] = [vehicle_location["latitude"], vehicle_location["longitude"]]

                # Parse package destination
                InputOutputTransformer.validateMessageForValue(parcel, ["destination"])
                destination = parcel.pop('destination')

                if payload["useCase"] == SLO_CRO_USE_CASE:
                    station_id = InputOutputTransformer.getStationIdOrClosest(destination,
                                                                              config_parser.get_csv_path(
                                                                                  use_case_graph), transformation_map,
                                                                              parcel_id)
                    parcel['destination'] = station_id
                else: #ELTA use_case
                    InputOutputTransformer.validateMessageForValue(destination, ["latitude", "longitude"])
                    parcel['destination'] = [destination["latitude"], destination["longitude"]]

                orders.append(parcel)

            ########################################################################################
            # GO THROUGH AVAILABLE VEHICLES AND PUT REMAINING PLAN LOAD ORDERS IN ARRAY OF PARCELS
            ########################################################################################
            #for remaining_clo in new_clos:
            #   orders.extend(
            #        InputOutputTransformer.updateOrdersList(remaining_clo, payload, parcels_dict, transformation_map))
            #
            ########################################################################################
            # REMAINING PLAN FOR BROKEN CLO (PICKUP PARCELS THAT BROKEN CLO SHOULD, BUT CANNOT)
            ########################################################################################
            #orders.extend(
            #    InputOutputTransformer.updateOrdersList(broken_clo, payload, parcels_dict, transformation_map))

            payload["orders"] = orders  # Orders that need to be processed by the remaining CLOs
        elif payload["eventType"] == "AdHocRequest":
            parcels = []
            for item in json["event"]["info"]["items"]:
                parcels.append(item)
            orders = InputOutputTransformer.buildOrdersFromParcels(parcels, payload["useCase"], use_case_graph, transformation_map)
            payload["eventType"] = "pickupRequest"  # setting back the event_type to basic use case
            payload["orders"] = orders
        else:
            parcels = []
            if json["parcels"] is not None:
                parcels = json["parcels"]
            orders = InputOutputTransformer.buildOrdersFromParcels(parcels, payload["useCase"], use_case_graph, transformation_map)
            payload["orders"] = orders

        return payload

    @staticmethod
    def updateOrdersList(clo, payload, parcels_dict, transformation_map):
        new_orders = []
        clo_state = clo["state"]

        if "remaining_plan" in clo_state and "steps" in clo_state["remaining_plan"]:
            for step in clo_state["remaining_plan"]["steps"]:
                InputOutputTransformer.validateMessageForValue(step, ["id"])
                if 'load' in step:
                    load = step["load"]
                    if load is None or len(load) == 0:
                        continue  # Skip, no parcels in vehicle plan

                    parcels = []
                    for parcel_id in load:
                        parcel_dict_cpy = copy.deepcopy(parcels_dict)
                        if parcel_id not in parcel_dict_cpy:
                            raise ValueError("Parcel with ID " + str(parcel_id) +
                                             " not on the list of 'parcels'!")
                        # Get parcel object
                        parcels.append(parcel_dict_cpy[parcel_id])

                    new_orders = InputOutputTransformer.buildOrdersFromParcels(
                        parcels, payload["useCase"], transformation_map)

        return new_orders

    @staticmethod
    def getStationIdOrClosest(location_dict, csv_file_path, transformation_map, parcel_id=None):
        station = None
        if "station" in location_dict:
            station = location_dict["station"]

        current_lat = None
        current_lon = None
        if station is None or station == "null" and ("latitude" in location_dict and "longitude" in location_dict):
            current_lat = location_dict["latitude"]
            current_lon = location_dict["longitude"]

            distances = {}  # Dict of distances between initial location and all the nodes in CSV file
            # Open file and read line by line
            with open(csv_file_path, encoding="utf8") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                for row in csv_reader:
                    id = row[1]
                    lat = row[2]
                    lon = row[3]

                    a = np.array((float(lat), float(lon)))
                    b = np.array((float(current_lat), float(current_lon)))

                    distances[id] = np.linalg.norm(a - b)

                csv_file.close()

                if parcel_id is not None:
                    transformation_map.map(parcel_id, [current_lat, current_lon])

                station = min(distances, key=distances.get)

        if station is None and (current_lat is None or current_lon is None):
            raise ValueError("location info should have a non-NULL fields 'station' "
                             "or 'latitude' and 'longitude'!")

        return station

    @staticmethod
    def buildOrdersFromParcels(parcels, use_case, use_case_graph, transformation_map):
        # Goes through parcels and builds an array of orders to be delivered
        orders = []

        for parcel in parcels:
            parcel['UUIDParcel'] = parcel.pop('id')
            parcel['weight'] = parcel.pop('payweight')

            InputOutputTransformer.validateMessageForValue(parcel, ["destination"])
            # Destination field is a JSON object
            destination = parcel.pop('destination')

            if use_case == SLO_CRO_USE_CASE:
                station_id = InputOutputTransformer.getStationIdOrClosest(
                    destination, config_parser.get_csv_path(use_case_graph), transformation_map)
                parcel['destination'] = station_id
            else:
                InputOutputTransformer.validateMessageForValue(destination, ["latitude", "longitude"])
                parcel['destination'] = [
                    destination["latitude"],
                    destination["longitude"]
                ]

            InputOutputTransformer.validateMessageForValue(parcel, ["source"])

            # Pickup field is a JSON object
            pickup = parcel.pop("source")
            if "country" in pickup:
                parcel['country'] = pickup["country"]

            if use_case == SLO_CRO_USE_CASE:
                station_id = InputOutputTransformer.getStationIdOrClosest(
                    pickup, config_parser.get_csv_path(use_case_graph), transformation_map, parcel["UUIDParcel"])
                parcel['pickup'] = station_id
            else:
                InputOutputTransformer.validateMessageForValue(pickup, ["latitude", "longitude"])
                parcel['pickup'] = [
                    pickup["latitude"],
                    pickup["longitude"]
                ]

            orders.append(parcel)

        return orders

    @staticmethod
    def incorrectFormatMessage(message):
        return {
            "message": message, "status": 0
        }

    @staticmethod
    def validateMessageForValue(field, keys):
        for key in keys:
            if key not in field or field[key] is None:
                raise ValueError(InputOutputTransformer.incorrectFormatMessage(
                    str(field) + " should have a non NULL field " + key))
        return None

    @staticmethod
    def prepare_output_message(recommendations, use_case, request_id, organization):
        # TODO: Replace location_id in messages to be in form of:
        # "location": { "latitude" : "xxx", "longitude": "xxx", "station": location_id}

        # recommendation_id ids a unique string for matching to response to the event for Angela
        recommendation_id = use_case + datetime.utcnow().strftime('%Y%m%d%H%M%S')
        clo_plans = []

        counter = 0
        for clo_plan in recommendations:
            counter += 1

            plan_organization = "ELTA"
            if use_case == SLO_CRO_USE_CASE:
                plan_organization = "PS" if clo_plan["UUID"].startswith("PS") else "HP"

            recommendation_text = "%srecommendation%s" % (use_case, counter)
            plan_id = "%splan%s" % (use_case, counter)
            message = {
                "clo": clo_plan["UUID"],  # Vehicle
                "id": recommendation_text,
                "plan": {
                    "id": plan_id,
                    "organization": plan_organization,
                    "execution_date": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    "recommendation": request_id,
                    "steps": clo_plan["route"]
                }
            }
            clo_plans.append(message)

        return {
            "organization": organization,
            "id": request_id,
            "cloplans": clo_plans
        }

    @staticmethod
    def revert_coordinates(recommendations, transformation_map):
        new_route_locations = []
        for recommendation in recommendations:
            last_id = 0
            if "route" in recommendation:
                routes = recommendation["route"]

                for route in routes:
                    last_id = int(route["id"])
                    load = route["load"]

                    for parcel_id in load:
                        if parcel_id in transformation_map.keys():
                            # Remove parcel from the 'load' and add it's ID with new locations in the dictionary
                            load.remove(parcel_id)
                            parcel_location_lat_lon = transformation_map.get(parcel_id)

                            added_to_array = False
                            for parcel_object in new_route_locations:
                                location = parcel_object.location

                                if location[0] == parcel_location_lat_lon[0] and location[1] == parcel_location_lat_lon[
                                    1]:
                                    parcel_object.addParcel(parcel_id)
                                    added_to_array = True
                                    continue
                            # Wasn't added to array, add new key and value
                            if not added_to_array:
                                parcel_loc = ParcelLocation(parcel_location_lat_lon, parcel_id)
                                new_route_locations.append(parcel_loc)

                for parcel_object in new_route_locations:
                    location = parcel_object.location

                    last_id = last_id + 1
                    routes.append({
                        "id": str(last_id),
                        "rank": str(last_id),
                        "complete": 0,
                        "due_time": None,
                        "load": parcel_object.parcels,  # Array of parcels
                        "unload": [],
                        "location": {'city': None, 'country': None, 'address': None,
                                     'latitude': location[0], 'longitude': location[1],
                                     'station': "station " + str(last_id), 'postal_code': None
                                     }
                    })

                recommendation["route"] = routes

        return recommendations

    @staticmethod
    def PickupNodeReorder(recommendations_raw):
        # 1. Itterate on vehicles.
        # 2. for each route -> find pickup nodes
        # 3. Reorder pickup nodes based on depencencies
        # 4. Delete pickup nodes from original route and start adding them based on ordered list
        # 5. For each pickup node: check where on route are nodes with parcels form pickup node = dependency locations
        # 6. Put pickup node on route before dependency locations
        # 7. Split the first part of the route and run TSP on it
        # 8. Itterate till Aall pickup nodes added

        """handling pickup locations nodes on route"""
        for i in range(len(recommendations_raw)):

            recommendations_new = []
            recommendations_new.append(copy.deepcopy(recommendations_raw[i]))
            route = copy.deepcopy(recommendations_new[0]["route"])
            PickupNodes = {}
            route_first_part = []
            route_second_part = []
            #create and ordered list of all pickup and dependency nodes
            final_route = []
            for station in route:
                if len(station["load"]) != 0:
                    PickupNodes[station["id"]] = station

            if len(PickupNodes) == 1: #if daily plan, skip reordering
                final_route = Tsp.order_recommendations(recommendations_new)[0]["route"]
            elif len(PickupNodes) >= 1:
                #create ordered list of pickup locations
                dependencies_stations = OrderRelations.create_relations(PickupNodes, route)

                #Removes the dependencies nodes withought loading

                PickupNodes_ids = [key for key, value in PickupNodes.items()]
                dependencies_list = []
                for idx, station in enumerate(dependencies_stations[:]):
                    if station["id"] in PickupNodes_ids:
                        dependencies_list.append(station)
                dependencies_stations = copy.deepcopy(dependencies_list)


                dependencies_stations_ids = [station["id"] for station in dependencies_stations]
                dependencies_stations.pop(0)
                dependencies_stations.reverse() #reverse ordered for adding to the the route

                #delete dependency stations from route + keep the start node + run TSP
                for station in route[1:]:
                    if station["id"] in dependencies_stations_ids:
                        route.remove(station)
                recommendations_new[0]["route"] = route
                recommendations_ordered = Tsp.order_recommendations(recommendations_new)
                route_first_part = recommendations_ordered[0]["route"]

                ##insert each pickup node before the parcel delivery stations
                for start_node in dependencies_stations:
                    pickupnode_parcels = set(start_node["load"])
                    station_idx = None
                    # check from second station on - the starting location does not change
                    for idx, station in enumerate(route_first_part[1:]):
                        if station_idx == None:
                            parcels = station["unload"]
                            for parcel in parcels:
                                if parcel in pickupnode_parcels:
                                    station_idx = idx + 1
                                    break

                    if station_idx == None:
                        print("adding nonconfict node to route_first_part")
                        route_first_part.append(start_node)
                        if len(route_first_part) >= 2:
                            recommendations_new[0]["route"] = route_first_part
                            route_tsp = Tsp.order_recommendations(recommendations_new)
                            route_first_part = route_tsp[0]["route"]
                        route_second_part2 = copy.deepcopy(route_first_part[1:])
                        route_second_part = route_second_part2 + route_second_part
                        route_first_part = copy.deepcopy(route_first_part[:1])
                    else:
                        print("adding  pickup node to route_first_part with conflict node")
                        route_second_part2 = (copy.deepcopy(route_first_part[station_idx:]))
                        route_first_part = copy.deepcopy(route_first_part[:station_idx])
                        route_second_part = route_second_part2 + route_second_part
                        route_first_part.append(start_node)
                        if len(route_first_part) >= 2:
                            recommendations_new[0]["route"] = route_first_part
                            route_tsp = Tsp.order_recommendations(recommendations_new)
                            route_first_part = route_tsp[0]["route"]
                final_route = route_first_part + route_second_part

            recommendations_raw[i]["route"] = copy.deepcopy(final_route)
        return recommendations_raw


    @staticmethod
    def PrintRoutes(recommendations):
        # Print route sequence for each CLO in recommendations plan.
        print("*******Route plan for delivery********")
        for plan in recommendations:
            print("Plan for VEHICLE:", plan["UUID"])
            string = ""
            for step in plan["route"]:
                location_name = (step["location"]["station"])
                loadparcels = ""
                for load in step["load"]:
                    loadparcels = loadparcels + " " + load
                unloadparcels = ""
                for unload in step["unload"]:
                    unloadparcels = unloadparcels + " " + unload

                step_string = "nodeName:" + " " + location_name + "Load:" + loadparcels
                print("STEP:", location_name)
                if loadparcels:
                    print("parcels loading:", loadparcels)
                if unloadparcels:
                    print("parcels unloading:", unloadparcels)
        print("****end of plan****")
        return(True)


class ParcelLocation:
    """
    Used for adding parcels to array which are meant for the same location.
    """

    def __init__(self, location, parcel):
        self.location = location
        self.parcels = [parcel]

    def addParcel(self, parcel):
        self.parcels.append(parcel)
