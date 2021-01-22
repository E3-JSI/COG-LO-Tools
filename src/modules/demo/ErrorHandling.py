import csv

from ..create_graph.config.config_parser import ConfigParser
import json
import datetime

class ErrorHandling:
    slo_case = {'SLO', 'PS', 'HP'}

    def __init__(self):
        self.config_parser = ConfigParser()

    def check_organization(self, input_data):
        '''
         Organization ce je slo-cro/ ps hp
        '''
        if input_data['event'] is None:
            return

        if input_data['organization'] == 'ELTA':
            if input_data['event']['event_type'] != 'vehicle':
                for item in input_data['event']['info']['items']:
                    if item['organization'] != 'ELTA':
                        raise ValueError("Wrong organization. Organization should be the same at the same message")

            for clo in input_data['clos']:
                if clo['info']['organization'] != 'ELTA':
                    raise ValueError("Wrong organization. Organization should be the same at the same message")

            for parcel in input_data['parcels']:
                if parcel['organization'] != 'ELTA':
                    raise ValueError("Wrong organization. Organization should be the same at the same message")

        if input_data['organization'] == 'PS':
            if input_data['event']['event_type'] != 'vehicle':
                for item in input_data['event']['info']['items']:
                    if item['organization'] not in self.slo_case:
                        raise ValueError("Wrong organization. Organization should be the same at the same message")

            for clo in input_data['clos']:
                if clo['info']['organization'] not in self.slo_case:
                    raise ValueError("Wrong organization. Organization should be the same at the same message")

            for parcel in input_data['parcels']:
                if parcel['organization'] not in self.slo_case:
                    raise ValueError("Wrong organization. Organization should be the same at the same message")

    def check_remaining_plan(self, input_data):
        '''
         daily plan -> complited plan prazn oz missing
        '''

        for clo in input_data['clos']:
            if 'remaining_plan' not in clo['state']:
                raise ValueError("Remaining_plan is missing in CLO")

    def check_locations(self, input_data):

        if (input_data['organization'] == "PS" or \
                input_data['organization'] == "HP") and (input_data["event"] == None):
            use_case_graph = ConfigParser()
            fObj = open(use_case_graph.get_csv_path("SLO-CRO_crossborder"))

            node_dict = {}
            with fObj as csvfile:
                nodes = csv.reader(csvfile, delimiter=',')
                for value in nodes:
                    node_dict[value[1]] = {'lat': float(value[2]), 'lon': float(value[3])}

            for clo in input_data['clos']:
                if clo['state']['location']['station'] not in node_dict or \
                        clo['state']['location']['latitude'] != node_dict[clo['state']['location']['station']]['lat'] or \
                        clo['state']['location']['longitude'] != node_dict[clo['state']['location']['station']]['lon']:
                    raise ValueError("Check vehicle lat, lon and uuid. It doesnt exists in our database {}"
                                     .format(clo['state']['location']))

            for clo in input_data['parcels']:
                if clo['source']['station'] not in node_dict or \
                        clo['source']['latitude'] != node_dict[clo['source']['station']]['lat'] or \
                        clo['source']['longitude'] != node_dict[clo['source']['station']]['lon']:
                    raise ValueError(
                        "Check Parcels lat, lon and uuid. It doesnt exists in our database {}".format(clo['source']), clo["id"])

                if clo['destination']['station'] not in node_dict or \
                        clo['destination']['latitude'] != node_dict[clo['destination']['station']]['lat'] or \
                        clo['destination']['longitude'] != node_dict[clo['destination']['station']]['lon']:
                    raise ValueError(
                        "Check Parcels lat, lon and uuid. It doesnt exists in our database {}".format(clo['destination'], clo["id"]))

                '''
                a = clo['state']['location']['latitude']
                b = node_dict[clo['state']['location']['station']]['lat']
                '''

        if input_data['organization'] == "SLO-CRO":
            pass

    def check_event(self, input_data):
        event_types = {'order', 'vehicle', 'border'}
        if input_data['event'] is None:
            return

        if 'event_type' not in input_data['event'] :
            raise ValueError("Event type is missing")

        if input_data['event']['event_type'] not in event_types:
            raise ValueError("Wrong event type.")

    def check_payweight(self, input_data):
        for clo in input_data['parcels']:
            print(clo)
            if 'payweight' not in clo or clo['payweight'] != 1:
                self.error = ValueError("Payweight is missing or not == 1.", clo["id"])
                raise self.error

    def check_parcel_clos(self, input_data):
        if 'parcels' not in input_data or \
                type(input_data['parcels']) is not list or \
                'clos' not in input_data or \
                type(input_data['clos']) is not list:
            raise ValueError("Parcels and ClOs wrong type.")

    def chech_parcel_id(self, input_data):
        l = []
        for clo in input_data['parcels']:
            l.append(clo["id"])
        if len(l) != len(set(l)):
            print("true")
            raise ValueError("Parcelid are not unique - duplicate id.", clo["id"])

    def write_file(self, input_data):
        date_time_obj = datetime.datetime.now()
        timestamp_str = date_time_obj.strftime("%d_%b_%Y_(%H_%M_%S)")
        logger_file = self.config_parser.get_logger_file()


        filename = logger_file + 'requests_' + timestamp_str + '.json'
        with open(filename, 'w') as outfile:
            json.dump(input_data, outfile)

    def check_messages_correction(self, input_data):

        self.write_file(input_data)

        self.check_event(input_data)
        self.check_organization(input_data)

        self.check_remaining_plan(input_data)

        self.check_locations(input_data)

        self.check_payweight(input_data)
        self.check_parcel_clos(input_data)

        self.chech_parcel_id(input_data)
