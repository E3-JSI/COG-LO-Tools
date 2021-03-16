import copy
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ..config.config_parser import ConfigParser
from ...utils.structures.deliveries import Deliveries
from ...utils.structures.parcel import Parcel
from ...utils.tsp import Tsp
from ...utils.input_output import InputOutputTransformer

config_parser = ConfigParser()
url = "https://graphhopper.com/api/1/vrp?key=e8a55308-9419-4814-81f1-6250efe25b5c"

#process Larissa use-case message
def orderELTA2Recommendations (recommendations, last_step, deliveries):
    ordered_recommendations = Tsp.order_recommendations(recommendations)
    ordered_recommendations[0]["route"].append(last_step)

    ordered_recommendations[0]["route"] = InputOutputTransformer.FirstStepProcessing(ordered_recommendations[0]["route"], deliveries)

    index = 1
    for step in ordered_recommendations[0]["route"]:
        step["id"] = index
        step["rank"] = index
        index +=1

    return ordered_recommendations


def create_ELTA_urban2_recommendations (data, deliveries):
    recommendations = []
    recommendation = {}
    start_address = {}
    route = []
    step_locations = []
    delivery_location = []

    recommendation["UUID"] = data["clos"][0]["UUID"]
    start_address["location_id"] = "vehicle_location"
    start_address["lat"] = data["clos"][0]["currentLocation"][0]
    start_address["lon"] = data["clos"][0]["currentLocation"][1]
    recommendation["start_address"] = start_address

    # steps for route plan
    index = 1
    for parcel in deliveries.req:
            # if more than one parcel on each step
        if parcel.current_location in step_locations:
            for step in route:
                if step["location"]["latitude"] == parcel.current_location[0] and step["location"]["longitude"] == \
                        parcel.current_location[1]:
                    step["load"].append(parcel.uuid)

        elif parcel.current_location not in step_locations:
            step_locations.append(parcel.current_location)

            step = {}
            location = {}
            dependency = {}

            location["address"] = "Address"
            location["city"] = None
            location["country"] = parcel.country
            location["latitude"] = parcel.current_location[0]
            location["longitude"] = parcel.current_location[1]
            location["postal_office"] = "Parcel Location" + str(index)
            location["station"] = "Parcel Location" + str(index)

            dependency["plan"] = None
            dependency["plan_step"] = None

            step["complete"] = 0
            step["due_time"] = None
            step["location"] = location
            step["dependency"] = dependency
            step["id"] = index
            step["rank"] = index
            step["load"] = [parcel.uuid]
            step["unload"] = []

            route.append(step)

        if parcel.target not in delivery_location:
            delivery_location.append(parcel.target)
        index +=1

    #add first step
    clo = data["clos"][0]
    if clo["currentLocation"] not in step_locations:
        first_step = {}
        location = {}
        dependency = {}

        location["address"] = "Address"
        location["city"] = None
        location["country"] = clo["country"]
        location["latitude"] = clo["currentLocation"][0]
        location["longitude"] = clo["currentLocation"][1]
        location["postal_office"] = "vehicle_location"
        location["station"] = "Vehicle Location"

        dependency["plan"] = None
        dependency["plan_step"] = None

        first_step["complete"] = 0
        first_step["due_time"] = None
        first_step["location"] = location
        first_step["dependency"] = dependency
        first_step["id"] = 0
        first_step["rank"] = 0
        first_step["load"] = []
        first_step["unload"] = []

    elif clo["currentLocation"] in step_locations:
        for idx, step in enumerate(route):
            if step["location"]["latitude"] == clo["currentLocation"][0] and step["location"]["longitude"] == \
                    clo["currentLocation"][1]:
                first_step = copy.deepcopy(step)
                route.pop(idx)

    #create last step
    parcel_ids = []
    last_step = {}
    location = {}
    dependency = {}

    for parcel in deliveries.req:
        parcel_ids.append(parcel.uuid)
    for parcel in deliveries.origin:
        parcel_ids.append(parcel.uuid)

    location["address"] = "Address"
    location["city"] = None
    location["country"] = clo["country"]
    location["latitude"] = delivery_location[0][0]
    location["longitude"] = delivery_location[0][1]
    location["postal_office"] = "Delivery Location"
    location["station"] = "LarissaDEPO"

    dependency["plan"] = None
    dependency["plan_step"] = None

    last_step["complete"] = 0
    last_step["due_time"] = None
    last_step["location"] = location
    last_step["dependency"] = dependency
    last_step["id"] = index
    last_step["rank"] = index
    last_step["load"] = []
    last_step["unload"] = parcel_ids

    #create recommendations
    route.insert(0, first_step)
    recommendation["route"] = route
    recommendations.append(recommendation)

    return recommendations, last_step


def create_ELTA_urban2_deliveries (clos, requests):
    deliveries_origin = []
    # list of additional parcels from request
    deliveries_diff = [Parcel(x["UUIDParcel"], x["destination"], x["weight"], x["pickup"], "order") for x in requests]
    # list of parcels on CLOs before request
    if len(clos) != 1:
        print("Incorrect number of vehicles, should be 1")
    for clo in clos:
        for parcel in clo["parcels"]:
            deliveries_origin.append(Parcel(parcel["UUIDParcel"], parcel["destination"],
                                            parcel["weight"], clo["currentLocation"], country=parcel["country"]))
    deliveries = Deliveries(deliveries_origin, deliveries_diff)

    return deliveries


# map parcels to exisitng virtual nodes

def map_coordinates_to_response(recommendations, transform_map_dict):
    for recommendation in recommendations:
        for route in recommendation['route'][:]:
            pickup_parcels = []
            if route['load'] != '' and len(route['load']):
                for route_parcel in route['load'][:]:
                    latitude = transform_map_dict[(route_parcel, 'pickup')][0]
                    longitude = transform_map_dict[(route_parcel, 'pickup')][1]
                    parcel_mapped = False
                    if len(pickup_parcels) != 0:
                        for parcel in pickup_parcels:
                            if latitude == parcel["latitude"] and longitude == parcel["longitude"]:
                                parcel["id"].extend([route_parcel])
                                parcel_mapped = True
                                break
                    if parcel_mapped == False:
                        pickup_parcels.append({
                            'id': [route_parcel],
                            'latitude': latitude,
                            'longitude': longitude
                        })
                route['load'] = pickup_parcels

            if route['unload'] != '' and len(route['unload']):
                delivery_parcels = []
                for route_parcel in route['unload']:
                    latitude = transform_map_dict[(route_parcel, 'destination')][0]
                    longitude = transform_map_dict[(route_parcel, 'destination')][1]
                    parcel_mapped = False
                    if len(delivery_parcels) != 0:
                        for parcel in delivery_parcels:
                            if latitude == parcel["latitude"] and longitude == parcel["longitude"]:
                                parcel["id"].extend([route_parcel])
                                parcel_mapped = True
                                break
                    if parcel_mapped == False:
                        delivery_parcels.append({
                            'id': [route_parcel],
                            'latitude': transform_map_dict[(route_parcel, 'destination')][0],
                            'longitude': transform_map_dict[(route_parcel, 'destination')][1]
                        })
                route['unload'] = delivery_parcels
    return recommendations


def get_orders_coordinates(data):
    transform_map_dict = {}
    if 'orders' in data:
        for i, el in enumerate(data['orders']):
            transform_map_dict[(el['UUIDParcel'], "destination")] = el['destination']
            transform_map_dict[(el['UUIDParcel'], "pickup")] = el['pickup']
        if 'clos' in data:
            for clos in data['clos']:
                for parcel in clos['parcels']:
                    transform_map_dict[(parcel['UUIDParcel'], "destination")] = parcel['destination']
                    transform_map_dict[(parcel['UUIDParcel'], "pickup")] = parcel['destination']
    elif 'brokenVehicle' in data:
        for clos in data['clos']:
            current_location = clos['currentLocation']
            for parcel in clos['parcels']:
                transform_map_dict[(parcel['UUIDParcel'], "destination")] = parcel['destination']
                transform_map_dict[(parcel['UUIDParcel'], "pickup")] = current_location

        current_location = data['brokenVehicle']['currentLocation']
        for parcel in data['brokenVehicle']['parcels']:
                transform_map_dict[(parcel['UUIDParcel'], "destination")] = parcel['destination']
                transform_map_dict[(parcel['UUIDParcel'], "pickup")] = current_location
    return transform_map_dict

def order_parcels_on_route(response, use_case_graph):
    if use_case_graph == "ELTA_urban2":
        return response

    step_number = 1
    for idx, clo in enumerate(response["cloplans"]):
        new_steps = []
        for step in clo['plan']["steps"]:
            if len(step['load']) != []:
                for parcel in step['load']:
                    new_step = copy.deepcopy(step)
                    new_step["id"] = step_number
                    new_step["load"] = parcel["id"]
                    new_step["location"]["latitude"] = parcel["latitude"]
                    new_step["location"]["longitude"] = parcel ["longitude"]
                    new_step["station"] = None
                    new_step["unload"] = []
                    new_steps.append(new_step)
                    step_number += 1

            if len(step['unload']) != []:
                for parcel in step['unload']:
                    new_step = copy.deepcopy(step)
                    new_step["id"] = step_number
                    new_step["unload"] = parcel["id"]
                    new_step["location"]["latitude"] = parcel["latitude"]
                    new_step["location"]["longitude"] = parcel["longitude"]
                    new_step["station"] = None
                    new_step["load"] = []
                    new_steps.append(new_step)
                    step_number += 1
        response["cloplans"][idx]["plan"]["steps"]=new_steps
    return response

def proccess_elta_event(proc_event, data, use_case_graph):
    if proc_event == None:
        return elta_clustering(data, use_case_graph)
    elif proc_event == 'pickupRequest' or proc_event == 'brokenVehicle':
        return elta_map_parcels(data, use_case_graph)


def find_min_elta(coords, clos_list):
    import sys
    from scipy.spatial import distance
    import haversine as hs
    cur_min = sys.maxsize
    label = -1
    lat, lon = coords
    for post in clos_list:
        lat_post = float(post["info"]["location"]["latitude"])
        lon_post = float(post["info"]["location"]["longitude"])
        distance = hs.haversine((lat, lon), (lat_post, lon_post))
        if distance < cur_min:
            cur_min = distance
            label = post["id"]
    if label == -1:
        raise Exception("Error in mapping parcels to nodes")
    return label


def elta_clustering(orig_data, use_case_graph):
    nClusters = 10
    data = copy.deepcopy(orig_data)
    l = []
    for i, el in enumerate(data['orders']):
        l.append([i, el['UUIDParcel'], "destination"] + el['destination'])
        #l.append([i, el['UUIDParcel'], "pickup"] + el['pickup'])
    df = pd.DataFrame(l)

        #skip clustering if number of parcels to small
    if (len(orig_data["orders"]) + 5) <= (nClusters):
        print("NO CLUSTERING - not enough parcels")
        clos = {"useCase": "ELTA"}
        clos_list = create_clo_list_elta_no_clusters(use_case_graph, data["orders"])

    else:  # run clustering
        kmeans = KMeans(n_clusters=nClusters)
        kmeans.fit(df[[3, 4]])  # Compute k-means clustering.
        centers = kmeans.cluster_centers_
        clos = {"useCase": "ELTA"}
        clos_list = create_clo_list_elta(use_case_graph, centers)

    #mapping locations to postal_Id
    for clo in data["clos"]:
        mapped_location_clo = find_min_elta(clo["currentLocation"], clos_list)
        clo["currentLocation"] = mapped_location_clo
    for order in data["orders"]:
        mapped_location_dest = find_min_elta(order["destination"], clos_list)
        order["destination"] = mapped_location_dest
        mapped_location_pickup = find_min_elta(order["pickup"], clos_list)
        order["pickup"] = mapped_location_pickup

        ## print clusters
        #df.plot.scatter(x=3, y=4, c=labels, s=10, cmap='viridis')
        #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=80, alpha=0.5)
        #plt.show()

    clos["clos"] = clos_list
    return data, clos


def create_clo_list_elta_no_clusters(use_case_graph, orders):
    clos_list = []
    i = 0
    for parcel in orders:
        lat, lon = parcel["destination"]
        clos_list.append({
            "id": str(i),
            "info": {
                "address": "virtualPost",
                "location": {
                    "latitude": lat,
                    "longitude": lon
                }
            }
        })
        i = i + 1

    elta = config_parser.get_elta_path(use_case_graph)
    with open(elta) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) != 0:
                clos_list.append({
                    "id": row[1],
                    "info": {
                        "address": row[0],
                        "location": {
                            "latitude": float(row[2]),
                            "longitude": float(row[3])
                        }
                    }
                })
    return clos_list


def create_clo_list_elta (use_case_graph, centers):
    clos_list = []
    i = 0
    for row in centers:
        clos_list.append({
            "id": str(i),
            "info": {
                "address": "virtualPost",
                "location": {
                    "latitude": row[0],
                    "longitude": row[1]
                }
            }
        })
        i = i + 1

    elta = config_parser.get_elta_path(use_case_graph)
    with open(elta) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) != 0:
                clos_list.append({
                    "id": row[1],
                    "info": {
                        "address": row[0],
                        "location": {
                            "latitude": float(row[2]),
                            "longitude": float(row[3])
                        }
                    }
                })
    return clos_list

def find_min(use_case_graph, lat_cord, lon_cord):
    elta = config_parser.get_csv_path(use_case_graph)
    posts = []
    with open(elta) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            posts.append([row[1], (row[2], row[3])])
    import sys
    cur_min = sys.maxsize
    label = -1
    for post in posts:
        lat = float(lat_cord) - float(post[1][0])
        lon = float(lon_cord) - float(post[1][1])
        distance = (lat * lat) + (lon * lon)
        if distance < cur_min:
            cur_min = distance
            label = post[0]
    if label == -1:
        raise Exception("Error in mapping parcels to nodes")

    return label

def elta_map_parcels(orig_data, use_case_graph):
    data = copy.deepcopy(orig_data)
    for clo in data['clos']:
        mapped_location = find_min(use_case_graph, clo['currentLocation'][0], clo['currentLocation'][1])
        clo['currentLocation'] = mapped_location
        clo['currentLocationL'] = clo['currentLocation']
        for parcel in clo['parcels']:
            mapped_location = find_min(use_case_graph, parcel['destination'][0], parcel['destination'][1])
            parcel['destination_location'] = parcel['destination']
            parcel['destination'] = mapped_location
    for order in data["orders"]:
        mapped_location = find_min(use_case_graph,order['destination'][0], order['destination'][1])
        order['destination_location'] = order['destination']
        order['destination'] = mapped_location
        mapped_location = find_min(use_case_graph, order['pickup'][0], order['pickup'][1])
        order['pickup_location'] = order['pickup']
        order['pickup'] = mapped_location

    return data
