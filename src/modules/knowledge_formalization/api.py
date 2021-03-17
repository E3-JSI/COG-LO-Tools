from waitress import serve

from flask import Flask
from flask import request

import time
import json

from db import Database
from formalization import validate_config
from ontology import OntologyInspector

# create Flask instance for an app

app = Flask(__name__)


@app.route('/getConceptRelationships', methods=['POST'])
def get_new_concepts():
    # received data must be in JSON format
    received_request = request.json
    if received_request == 'null' or received_request is None:
        return 'Error parsing request. It should be in JSON format'

    # Return response, because we received request with ID for result query.
    if 'id' in received_request:
        id = received_request['id']
        if not isinstance(id, int):
            return 'id parameter should be an integer.'

        result = database.query("SELECT result FROM concepts WHERE id = %s;", (id,))
        if result is None or not result:
            return {'result': 'null', 'id': id}
        else:
            return {'result': result[0], 'id': id}

    if 'alpha' not in received_request:
        return 'JSON object in request must contain key "alpha".'

    if 'number' not in received_request and 'percentage' not in received_request:
        return 'JSON object in request must contain key "number" or "percentage", ' \
               'representing number of new concepts wanted or percentage of new concepts respectfully.'

    if 'number' in received_request and 'percentage' in received_request:
        return 'Request must contain only key "number" or "percentage", not both.'

    # parameter for retrieving concepts can be given as number of new concepts or as percentage of new
    # concepts in respect to the number of all concepts which is 13500000
    if 'number' in received_request:
        # number value must be an integer
        value = received_request['number']
        if not isinstance(value, int):
            return 'Number parameter should be an integer.'
        if value > 600:
            return 'Number value cannot be larger than 600.'
    else:
        # percentage value can be given as integer or a float
        value = received_request['percentage']
        if not isinstance(value, int) and not isinstance(value, float):
            return 'Percentage parameter should be an integer or a float.'
        if value > 0.005:
            return 'Percentage value cannot be larger than 0.005.'
        value = (value / 100.0) * 13500000

    # extract alpha parameter from request
    alpha = received_request['alpha']
    if alpha <= 0 or alpha > 1:
        return 'Alpha parameter must be between 0 and 1.'

    if value <= 0:
        return "Number of new concepts must be more than 0."

    # get current timestamp
    timestamp = int(round(time.time()))
    # round value to integer
    value = int(round(value))

    concepts_array = None
    if 'concepts' in received_request:
        payload = received_request['concepts']
        concepts_array = json.dumps(payload)

    # store request to the database and return ID of inserted row
    row_id = database.execute("INSERT INTO concepts (id, timestamp, alpha, concepts_number, concepts) "
                              "VALUES (DEFAULT, %s, %s, %s, %s) RETURNING id;", (timestamp, alpha, value, concepts_array), True)
    # error inserting into database
    if row_id is None:
        return "Error processing/storing request into the database."

    # create response as json object
    response = {'stored': True, 'processed': False, 'id': row_id}

    return response

@app.route('/initializeKnowledgeBase', methods=['POST'])
def generate_new_initial_concepts():
    received_request = request.json
    if received_request == 'null' or received_request is None:
        return 'Error parsing request. It should be in JSON format'

    # request should contain 'concepts' key
    if 'concepts' not in received_request:
        return 'JSON object in request must contain key "concepts".'

    # extract payload from concepts
    payload = received_request['concepts']

    # value of key 'concepts' must be an array
    if not isinstance(payload, list) or not payload:
        return 'Value of key concepts should be non empty array'

    # map unicode words to strings
    # payload = map(str, payload)

    # store payload as json object
    json_obj = json.dumps(payload)

    # get current timestamp
    timestamp = int(round(time.time()))
    # store to the database and return row_id
    database.execute("INSERT INTO initial_concepts (id, timestamp, processed, concepts) "
                              "VALUES (DEFAULT, %s, FALSE, %s) RETURNING id;", (timestamp, json_obj), True)

    return {'success': True}

@app.route('/getConceptMappings', methods=['POST'])
def get_concept_mappings():
    """
    Request should look like: {"concepts": ["Station", "ExchangeHub"]}
    :return:
    """
    received_request = request.json
    if received_request == 'null' or received_request is None:
        return 'Error parsing request. It should be in JSON format'

    if 'concepts' not in received_request:
        return 'JSON object in request must contain key "concepts".'

    # extract payload from concepts
    payload = received_request['concepts']

    # value of key 'concepts' must be an array
    if not isinstance(payload, list) or not payload:
        return 'Value of key concepts should be non empty array'

    # result example:
    # {
    #     "results": [
    #         {
    #             "concept": "TransportationMeans",
    #             "query_concept": "Truck",
    #             "relationship": "type"
    #         }
    #     ]
    # }
    results = ontology_inspector.get_concept_mappings(payload)
    return {'results': results}


@app.route('/getOntology', methods=['POST'])
def get_ontology():
    return ontology_inspector.get_ontology_json()


if __name__ == '__main__':
    # read config, validate it and extract database config
    config_file_path = "./config/config.json"
    config = validate_config(config_file_path)
    cfg = config["database"]

    # get instance of database connection
    db_name = "concepts_db"
    database = Database(db_name, cfg)
    resources = config["resources"]

    # create instance of Ontology inspector
    ontology_path = resources["ontology_path"]
    ontology_inspector = OntologyInspector(ontology_path)

    # start API
    serve(app, host='localhost', port=5555)
