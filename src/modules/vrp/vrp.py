import matlab.engine

from waitress import serve
from flask import request
from flask import Flask

import time
import json


class VRP:
    def __init__(self):
        # initialize MATLAB
        print('initializing the MATLAB engine')

        self.engine = matlab.engine.start_matlab()
        self.engine.cd('modules/vrp/matlab')
        self.engine.addpath('./utils')

    def vrp(self, req_json):
        print("received VRP request with content: ", req_json)

        incidence_mat = req_json['incidenceMat']
        cost_mat = req_json['costMat']
        edge_time_vec = req_json['edgeTimeV']
        start_vec = req_json['startV']
        end_vec = req_json['endV']
        distr_vec = req_json['nodeDistributionV']
        vehicle_capacity_vec = req_json['vehicleCapacityV']
        t_start_vec = req_json['nodeOpenV']
        t_end_vec = req_json['nodeCloseV']

        E = matlab.double(incidence_mat)
        C = matlab.double(cost_mat)
        t_vec = matlab.double(edge_time_vec)
        start_v = matlab.double(start_vec)
        end_v = matlab.double(end_vec)
        distr_v = matlab.double(distr_vec)
        capacity_v = matlab.double(vehicle_capacity_vec)
        t_start_v = matlab.double(t_start_vec)
        t_end_v = matlab.double(t_end_vec)

        start_tm = time.time()

        R, cost = self.engine.vrptw_solve(
            E,
            C,
            t_vec,
            start_v,
            end_v,
            distr_v,
            capacity_v,
            t_start_v,
            t_end_v,
            nargout=2
        )

        end_tm = time.time()

        n_rows = R.size[0]
        n_cols = R.size[1]

        R_py = [[R[rowN][colN] for colN in range(n_cols)] for rowN in range(n_rows)]

        print('request processed in ' + str(round(end_tm - start_tm, 2)) + ' seconds')

        # Return computed routes (R_py) and costs (cost)
        return R_py, cost


if __name__ == '__main__':

    print('initializing Flask')
    app = Flask(__name__)
    vrp = VRP()

    @app.route('/api/vrptw', methods=['POST'])
    def callVrptw():
        print('processing VRPTW request')
        req_json = request.json

        if req_json is None or req_json == 'null':
            return 'Error parsing request! Should be in JSON format!'

        routes, cost = vrp.vrp(req_json)

        return json.dumps({
            'routes': routes,
            'cost': cost
        })

    serve(app, host='localhost', port=4504)
