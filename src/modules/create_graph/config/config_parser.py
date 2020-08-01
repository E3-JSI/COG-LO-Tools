import os
import json


class ConfigParser:

    def __init__(self, config_path='./modules/create_graph/config/config.json'):
        if not os.path.exists(config_path):
            print("File config.json does not exist. Config parser cannot be initialized!")
            exit(1)

        with open(config_path) as config:
            self.json_config = json.load(config)
        config.close()

    def get_pickle_path(self, use_case):
        if use_case == "SLO-CRO":
            return self.json_config["slo_cro_pickle_path"]
        elif use_case == "ELTA":
            return self.json_config["elta_pickle_path"]

    def get_graph_path(self, use_case):
        if use_case == "SLO-CRO":
            return self.json_config["slo_cro_json_graph_data_path"]
        elif use_case == "ELTA":
            return self.json_config["elta_json_graph_data_path"]
        else:
            print("Error - use_case not defined.")

    def get_csv_path(self, use_case):
        if use_case == "SLO-CRO":
            return self.json_config["post_loc_slo_cro"]
        elif use_case == "ELTA":
            return self.json_config["post_loc_elta"]
        else:
            print("Error - use not not defined.")

    def get_border_nodes_slo(self):
        return self.json_config["slo_border_nodes"]

    def get_border_nodes_cro(self):
         return self.json_config["cro_border_nodes"]

    def get_msb_few_url(self):
        return self.json_config["msb_fwd"]

    def get_basic_map(self):
        return self.json_config["map_basic"]

    def get_eps(self):
        return self.json_config["eps"]

    def get_post_loc_type(self, use_case):
        if use_case == "SLO-CRO":
            return self.json_config["post_loc_type_slo_cro"]
        elif use_case == "ELTA":
            return self.json_config["post_loc_type_elta"]
        else:
            print("Error - use not not defined.")

    def get_slo_graph_path(self):
        # Get path of SLO graph
        return self.json_config["slo_graph_json_path"]

    def get_cro_graph_path(self):
        # Get path of CRO graph
        return self.json_config["cro_graph_json_path"]

    def get_graph_partitions(self):
        # Get the number of partitions used for graph split by partitioner
        return self.json_config["graph_partitions"]

    def get_slo_pickle_path(self):
        # Get path of CRO graph pickle
        return self.json_config["slo_graph_pickle_path"]

    def get_cro_pickle_path(self):
        # Get path of SLO graph pickle
        return self.json_config["cro_graph_pickle_path"]
