import os
import csv


class CloUpdateHandler:
    """
    Class used for handling new received CLOs and updating or recreating CSV file
    containing CLO data.
    """
    @staticmethod
    def extract_received_and_stored_dict(clos, csv_file_path):
        """
        Extract two dictionaries (received and stored post offices)
        """
        stored_clos_uuid_map = {}

        # Create map of received CLO UUIDs with content (latitude, longitude, address)
        received_clos_uuid_map = {}
        for json_obj in clos:
            uuid = json_obj["uuid"]
            received_clos_uuid_map[uuid] = json_obj

        if not os.path.isfile(csv_file_path):
            print("Postal offices csv file does not exist yet.")
        else:
            # Open file and read line by line
            stored_clos_uuid_map = {}
            with open(csv_file_path, encoding="utf8") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                for row in csv_reader:
                    # Get all important values - UUID, address, latitude and longitude
                    address = row[0]
                    uuid = row[1]
                    lat = row[2]
                    lon = row[3]

                    # Map of stored CLOs
                    stored_clos_uuid_map[uuid] = {
                        "uuid": uuid,
                        "lat": lat,
                        "lon": lon,
                        "address": address
                    }
            csv_file.close()

        return received_clos_uuid_map, stored_clos_uuid_map

    @staticmethod
    def handle_new_clo_request(clos, csv_file_path):

        """
               Handle request for new CLOs. If any of received post offices is not stored in our file from which graph is
               built, we will rewrite the file and rebuild the graph.
               :param clos:
               :param csv_file_path:
               :return:

               Example request (field clos):
               {
                   "CLOS": [
                       {
                           "uuid": "1212",
                           "lat": "14.12222",
                           "lon": "47.41243124",
                           "address": "Test road"
                       },
                        {
                           "uuid": "1214",
                           "lat": "14.5235235",
                           "lon": "46.521424",
                           "address": "Hoolywood road"
                        }
                   ]
               }
               """
        received_clos_uuid_map, stored_clos_uuid_map = CloUpdateHandler.extract_received_and_stored_dict(clos,
                                                                                                         csv_file_path)
        build_new_graph = False

        if len(received_clos_uuid_map) != len(stored_clos_uuid_map):
            build_new_graph = True

        else:
            # Go through received CLOs and check if we do not have all of them stored
            for received_key in received_clos_uuid_map.keys():
                received_object = received_clos_uuid_map[received_key]

                # Check if there is difference in uuids
                if received_key not in stored_clos_uuid_map.keys():
                    #stored_clos_uuid_map.append()
                    build_new_graph = True
                    break
                #check is there is difference in other values
                else:
                    stored_object = stored_clos_uuid_map[received_key]
                    # Something is different, update this entry
                    if stored_object["lat"] != received_object["lat"] or stored_object["lon"] != \
                            received_object["lon"] or stored_object["address"] != received_object["address"]:
                        build_new_graph = True
                        break

        if build_new_graph:
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)

                for json_obj in clos:
                    csv_writer.writerow([json_obj["address"], json_obj["uuid"], json_obj["lat"], json_obj["lon"]])
            csv_file.close()

        return build_new_graph

    @staticmethod
    def handle_update_clo_request(clos, csv_file_path):
        """
        Method used for detecting which CLO's changed and how.
        :param clos:
        :param csv_file_path:
        :return:

        Example request (field clos):
        {
            "CLOS": [
                {
                    "uuid": "1212",
                    "lat": "14.12224442",
                    "lon": "47.41243121445",
                    "address": "Miami, Florida",
                    "action": "update"
                },
                 {
                    "uuid": "1214",
                    "lat": "14.5235235",
                    "lon": "46.521424",
                    "address": "Hoolywood road",
                    "action": "remove"
                 }
            ]
        }
        """
        received_clos_uuid_map, stored_clos_uuid_map = CloUpdateHandler.extract_received_and_stored_dict(clos, csv_file_path)
        build_new_graph = False
        clos_to_add_dict = {}

        # Go through received CLOs and check if we do not have all of them stored
        for received_key in received_clos_uuid_map.keys():
            received_object = received_clos_uuid_map[received_key]
            if "action" not in received_object:
                received_object["action"] = None
            action = received_object["action"]
            received_object.pop("action", None)

            if received_key not in stored_clos_uuid_map.keys():
                if action is None or action == "add" or action == "update":
                    print("received object", received_object)

                    clos_to_add_dict[received_key] = received_object
                    build_new_graph = True # At least one CLO needs to be added or updated
            else:
                # We have stored entry for this key
                stored_object = stored_clos_uuid_map[received_key]

                # If "remove", just continue with for loop
                if action == "remove":
                    build_new_graph = True
                    continue
                elif stored_object["lat"] != received_object["lat"] or stored_object["lon"] != \
                            received_object["lon"] or stored_object["address"] != received_object["address"]:
                    build_new_graph = True

                    # Just add already stored object
                    clos_to_add_dict[received_key] = stored_object

        if build_new_graph:
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)

                for key in clos_to_add_dict.keys():
                    obj = clos_to_add_dict[key]
                    csv_writer.writerow([obj["address"], obj["uuid"], obj["lat"], obj["lon"]])
            csv_file.close()

        return build_new_graph
