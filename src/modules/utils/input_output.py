from datetime import datetime
from ..create_graph.config.config_parser import ConfigParser
from ..utils.clo_update_handler import CloUpdateHandler

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
    def clo_info(organization, physical_id, type, subtype, description, location, capacity, capacity_unit, working_hours):
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
    def transportation_state(location, status, prediction, load_volume, available, capacity_unit, driver, plan, completed, remaining, delay):
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
    def parcel(organization, id, source_id, source_type, source_location, destination_id, destination_type, destination_location, service_type, payweight, issued, customs):
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
    def parse_received_recommendation_message(json):
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
                        type = "pickupRequest"
                    elif type == "vehicle":
                        type = "brokenVehicle"
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
                        type = "pickupRequest"
                    elif type == "vehicle":
                        type = "brokenVehicle"
                    payload["eventType"] = type

        InputOutputTransformer.validateMessageForValue(json, ["clos"])
        clos = json["clos"]

        for clo in clos:
            InputOutputTransformer.validateMessageForValue(clo, ["id"])
            clo["UUID"] = clo["id"]

            location = None
            if "info" in clo:
                if "location" in clo["info"]:
                    location = clo["info"].pop('location')

                    if location["latitude"] is None or location["longitude"] is None:
                        location = None

            if location is None and "state" in clo:
                if "location" in clo["state"]:
                    location = clo["state"].pop('location')

            InputOutputTransformer.validateMessageForValue(location, ["latitude", "longitude"])
            if payload["useCase"] == SLO_CRO_USE_CASE:
                # Extract station ID from given 'latitude' and 'longitude'.
                csv_file = config_parser.get_csv_path(SLO_CRO_USE_CASE)
                location_station_dict = CloUpdateHandler.extract_location_station_dict(csv_file)
                station_id = location_station_dict[(str(location["latitude"]), str(location["longitude"]))]

                clo['currentLocation'] = station_id
            else:
                clo['currentLocation'] = [
                    location["latitude"],
                    location["longitude"]
                ]
            if "country" in location:
                clo['country'] = location["country"]

            InputOutputTransformer.validateMessageForValue(clo, ["info"])
            InputOutputTransformer.validateMessageForValue(clo["info"], ["capacity"])
            clo['capacity'] = clo["info"].pop('capacity')

            # Check for state -> remaining_plan -> steps
            if "state" not in clo:
                clo["parcels"] = []
                continue
            if clo["state"] is None:
                clo["parcels"] = []
                continue
            if "remaining_plan" not in clo["state"]:
                clo["parcels"] = []
                continue
            if "steps" not in clo["state"]["remaining_plan"]:
                clo["parcels"] = []
                continue

            parcels = clo["state"]["remaining_plan"]["steps"]
            clo_parcels = []

            for parcel in parcels:
                InputOutputTransformer.validateMessageForValue(parcel, ["id"])
                InputOutputTransformer.validateMessageForValue(parcel, ["payweight"])

                parcel['UUIDParcel'] = parcel.pop('id')
                parcel['weight'] = parcel.pop('payweight')

                InputOutputTransformer.validateMessageForValue(parcel, ["location"])
                destination = parcel.pop('location')

                if payload["useCase"] == SLO_CRO_USE_CASE:
                    InputOutputTransformer.validateMessageForValue(destination, ["station"])
                    parcel['destination'] = destination["station"]
                else:
                    InputOutputTransformer.validateMessageForValue(destination, ["latitude", "longitude"])
                    parcel['destination'] = [
                        destination["latitude"],
                        destination["longitude"]
                    ]
                if "country" in destination:
                    parcel['country'] = destination["country"]

                clo_parcels.append(parcel)

            clo["parcels"] = clo_parcels
        payload["clos"] = clos

        # TODO: This needs to be tested furthermore on examples provided by testers!!!
        if payload["eventType"] == "brokenVehicle":
            InputOutputTransformer.validateMessageForValue(json["event"], ["info"])
            InputOutputTransformer.validateMessageForValue(json["event"], ["info"])
            InputOutputTransformer.validateMessageForValue(json["event"]["info"], ["clo"])
            broken_vehicle = json["event"]["info"]["clo"]

            current_location = None
            if "info" in broken_vehicle:
                if "location" in broken_vehicle["info"]:
                    current_location = broken_vehicle["info"].pop('location')

                    if current_location["latitude"] is None or current_location["longitude"] is None:
                        current_location = None

            if current_location is None and "state" in broken_vehicle:
                if "location" in broken_vehicle["state"]:
                    current_location = broken_vehicle["state"].pop('location')

            if payload["useCase"] == SLO_CRO_USE_CASE:
                InputOutputTransformer.validateMessageForValue(current_location, ["station"])
                broken_vehicle["currentLocation"] = current_location["station"]
            else:
                InputOutputTransformer.validateMessageForValue(current_location, ["latitude", "longitude"])
                broken_vehicle['currentLocation'] = [
                    current_location["latitude"],
                    current_location["longitude"]
                ]
            if "country" in current_location:
                broken_vehicle['country'] = current_location["country"]

            InputOutputTransformer.validateMessageForValue(broken_vehicle, ["state"])
            state = broken_vehicle["state"]

            InputOutputTransformer.validateMessageForValue(state, ["parcels"])
            parcels = []
            for parcel in state["parcels"]:
                parcel['UUIDParcel'] = parcel.pop('id')
                parcel['weight'] = parcel.pop('payweight')

                InputOutputTransformer.validateMessageForValue(parcel, ["destination"])
                destination = parcel.pop('destination')

                if payload["useCase"] == SLO_CRO_USE_CASE:
                    InputOutputTransformer.validateMessageForValue(destination, ["station"])
                    parcel['destination'] = destination["station"]
                else:
                    InputOutputTransformer.validateMessageForValue(destination, ["latitude", "longitude"])
                    parcel['destination'] = [
                        destination["latitude"],
                        destination["longitude"]
                    ]
                if "country" in destination:
                    parcel['country'] = destination["country"]

                # Append as last element
                parcels.append(parcel)

            broken_vehicle["parcels"] = parcels
            payload["brokenVehicle"] = broken_vehicle
        else:
            orders = []
            if json["parcels"] is not None:
                orders = json["parcels"]

            for parcel in orders:
                parcel['UUIDParcel'] = parcel.pop('id')
                parcel['weight'] = parcel.pop('payweight')

                InputOutputTransformer.validateMessageForValue(parcel, ["destination"])
                # Destination field is a JSON object
                destination = parcel.pop('destination')

                if payload["useCase"] == SLO_CRO_USE_CASE:
                    InputOutputTransformer.validateMessageForValue(destination, ["station"])
                    parcel['destination'] = destination["station"]
                else:
                    InputOutputTransformer.validateMessageForValue(destination, ["latitude", "longitude"])
                    parcel['destination'] = [
                        destination["latitude"],
                        destination["longitude"]
                    ]
                if "country" in destination:
                    parcel['country'] = destination["country"]

                InputOutputTransformer.validateMessageForValue(parcel, ["source"])
                # Pickup field is a JSON object
                pickup = parcel.pop("source")

                if payload["useCase"] == SLO_CRO_USE_CASE:
                    InputOutputTransformer.validateMessageForValue(pickup, ["station"])
                    parcel['pickup'] = pickup["station"]
                else:
                    InputOutputTransformer.validateMessageForValue(pickup, ["latitude", "longitude"])
                    parcel['pickup'] = [
                      pickup["latitude"],
                      pickup["longitude"]
                    ]
                if "country" in pickup:
                    parcel['country'] = pickup["country"]
            payload["orders"] = orders

        return payload

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
    def prepare_output_message(recommendations, use_case, request_id):
        # TODO: Replace location_id in messages to be in form of:
        # "location": { "latitude" : "xxx", "longitude": "xxx", "station": location_id}

        # recommendation_id ids a unique string for matching to response to the event for Angela
        recommendation_id = use_case + datetime.utcnow().strftime('%Y%m%d%H%M%S')
        clo_plans = []

        counter = 0
        for clo_plan in recommendations:
            counter+=1
            recommendation_text = "%srecommendation%s" % (use_case, counter)
            plan_id = "%splan%s" % (use_case, counter)
            message = {
                "clo": clo_plan["UUID"], # Vehicle
                "id": recommendation_text,
                "plan": {
                    "id": plan_id,
                    "organization": use_case,
                    "execution_date": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    "recommendation": recommendation_id,
                    "steps": clo_plan["route"]
                }
            }
            clo_plans.append(message)

        return {
            "organization": use_case,
            "id": recommendation_id,
            "cloplans": clo_plans
        }
