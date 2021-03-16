import matlab.engine

from waitress import serve
from flask import request
from flask import Flask

import time

import json

if __name__ == '__main__':

    app = Flask(__name__)

    # initialize MATLAB
    engine = matlab.engine.start_matlab()
    engine.cd('./matlab')
    engine.addpath('./utils')

    @app.route('/api/vrptw', methods=['POST'])
    def callVrptw():
        req_json = request.json

        if req_json is None or req_json == 'null':
            return 'Error parsing request! Should be in JSON format!'

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

        R, cost = engine.vrptw_solve(
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

        return json.dumps({
            'routes': R_py,
            'cost': cost
        })


    serve(app, host='localhost', port=4504)