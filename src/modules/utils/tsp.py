import requests
import json

# URL to our TSP service which runs on the same machine as COGLO services, but a different port.
TSP_URL = "http://localhost:1234/api/tsp/"


# Travelling Salesman Problem algorithm which receives a list of 'routes' and returns
# ordered list of visits as a solution to TSP.
class Tsp:
    @staticmethod
    def order_recommendations(recommendations):
        dictionary, vehicle_array = Tsp.build_location_id_route_dict(recommendations)
        built_request = Tsp.build_input_request(recommendations)
        tsp_response, tsp_success = Tsp.send_request(built_request)

        if tsp_success:
            return Tsp.parse_response(tsp_response, dictionary, vehicle_array)
        else:
            print("TSP algorithm execution failed. Returning default 'recommendations' calculated without applying TSP")
            # Just return default recommendations
            return recommendations

    @staticmethod
    def build_input_request(recommendations):
        """
        Builds input request from calculated recommendations.
        :param recommendations:
        :return:
        """
        requests_array = []
        for recommendation in recommendations:
            vehicles_array = []
            services_array = []

            if "UUID" not in recommendation or "start_address" not in recommendation or "route" not in recommendation:
                continue

            vehicle_id = recommendation["UUID"]
            start_address = recommendation["start_address"]
            route = recommendation["route"]

            vehicles_array.append({
                "vehicle_id": vehicle_id,
                "start_address": start_address
            })

            for route in route:
                if "location" not in route:
                    continue
                location = route["location"]

                services_array.append({
                    "id": location["location_id"],
                    "address": {
                        "lat": location["latitude"],
                        "lon": location["longitude"],
                        "location_id": location["location_id"]
                    }
                })

            requests_array.append({
                "vehicles": vehicles_array,
                "services": services_array
            })

        return requests_array

    @staticmethod
    def build_location_id_route_dict(recommendations):
        """
        Builds dictionary from location_id to route plan and array of recommendations (plans for each vehicle)
        :return:
        """
        dictionary = [{}]
        recommendations_array = []

        for index in range(len(recommendations)):
            recommendation = recommendations[index]
            dictionary.append({})

            routes = recommendation["route"]
            vehicle_id = recommendation["UUID"]
            recommendations_array.append(vehicle_id)

            for route in routes:
                if "location" not in route:
                    continue
                location = route["location"]
                dictionary[index][location["location_id"]] = route

        return dictionary, recommendations_array

    @staticmethod
    def parse_response(tsp_recommendations, mapping, vehicle_array):
        """
        Parses response by TSP to work with existing data structures.
        :param tsp_recommendations:
        :param mapping:
        :param vehicle_array:
        :return:
        """
        recommendations = []
        for index in range(len(tsp_recommendations)):
            response = tsp_recommendations[index]

            # Cast to JSON object
            json_response = json.loads(response)

            # extract vehicle_id
            vehicle_id = vehicle_array[index]

            # Extract array of visits
            visits = json_response["visits"]

            routes = []
            for visit in visits[1:]:  # Skips first element, because TSP seems to have a bug and duplicates first entry
                # From "id" fields in visits we need to retrieve "route" that is stored in a dictionary
                route = mapping[index][visit["id"]]
                routes.append(route)

            recommendations.append({
                "UUID": vehicle_id,
                "route": routes
            })

        return recommendations

    @staticmethod
    def send_request(request_array):
        """
        Request example:
        {
            "vehicles": [
                {
                    "vehicle_id": "S1",
                    "start_address": {
                        "location_id": "Kapele",
                        "lon": 15.6780936,
                        "lat": 45.9310499
                    }
                }
            ],
            "services": [
                {
                    "id": "S4",
                    "name": "coglo",
                    "address": {
                        "location_id": "Topliska cesta",
                        "lon": 15.6214561,
                        "lat": 45.89337
                    }
                },
                {
                    "id": "S8",
                    "name": "coglo",
                    "address": {
                        "location_id": "Ulica stare pravde",
                        "lon": 15.5934382,
                        "lat": 45.9046721
                    }
                },
                {
                    "id": "S6",
                    "name": "coglo",
                    "address": {
                        "location_id": "Ulica 11. novembra",
                        "lon": 15.4781109,
                        "lat": 45.9419906
                    }
                }
            ]
        }

        Response example:
        {
            "duration_seconds": 1980,
            "visits": [
                {
                    "id": "Kapele"
                },
                {
                    "id": "Ulica stare pravde"
                },
                {
                    "id": "Topliska cesta"
                },
                {
                    "id": "Ulica 11. novembra"
                }
            ],
            "distance_meters": 31784
        }

        :param request_array: array of requests to be processed by TSP
        :return:
        """

        transform_success = False
        response = []
        for index in range(len(request_array)):
            print("sending TSP request for plan", index)
            request = request_array[index]

            headers = {'Content-type': 'application/json'}
            try:
                tsp_solution = requests.post(TSP_URL, json=request, headers=headers).text
                response.append(tsp_solution)
                transform_success = True
                print("received TSP response for plan", index)
            except Exception as ex:
                print("Error occurred sending request to TSP service", ex)
                transform_success = False

        return response, transform_success
