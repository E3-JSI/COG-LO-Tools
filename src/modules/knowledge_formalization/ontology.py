import subprocess
import json
from os import path

class OntologyInspector:

    path = None
    ontology = None
    edges = None

    def __init__(self, path):
        self.path = path
        self.edges = []
        self.ontology = None
        self.initOntology()

    def initOntology(self):
        """
        Transform owl ontology file to a json using robot.jar
        :return:
        """
        json_file = "ontology.json"

        if path.exists(json_file):
            print("parsed ontology file exists, skipping conversion from OWL to JSON")
        else:
            try:
                cmd = '-jar robot.jar convert --input ' + self.path + " --output " + json_file
                subprocess.call('java ' + cmd, shell=True)
            except:
                print("cannot transform OWL file to JSON using robot.jar")
                return

            try:
                # TODO: Try replacing hardcoded_string with something generic, like a ontology signature (web path)
                # it doesn't work at the time, because semanticweb.org website is down

                # clear file, remove everything before # character in a file and store it in the same file (json_file)
                hardcoded_string = "http://www.semanticweb.org/angela/ontologies/2019/5/COG-LO-ontology#"
                raw_file = open(json_file, 'r')
                line_list = raw_file.readlines()
                raw_file.close()

                with open(json_file, 'w') as modified_file:
                    for line in line_list:
                        new_line = line
                        if hardcoded_string in line:
                            new_line = line.replace(hardcoded_string, "")
                        modified_file.write(new_line)
                modified_file.close()
            except:
                print("cannot remove hardcoded strings from ontology in json format")
                return

        # open file as JSON
        with open(json_file) as json_file:
            data = json.load(json_file)
            self.ontology = data
            graph = data["graphs"]
            content = graph[0]

            # We are only interested in edges.
            # Example: Timetable (sub) is_a PhysicalEntity (obj)
            # Example: Driver (sub) is_a HumanOperator (obj)
            # Example of JSON edge:
            #     {
            #       "sub" : "ExchangeHub",
            #       "pred" : "type",
            #       "obj" : "Station"
            #     }
            self.edges = content["edges"]  # used for retrieving concept mapping
        json_file.close()
        print("finished parsing ontology")

    def get_concept_mappings(self, payload):
        """
        Payload should be an array of concepts, for example ["ship", "truck"]
        :param payload:
        :return:
        """

        # Ontology wasn't parsed correctly, just return empty array
        if self.ontology is None:
            return []

        result = []
        for query_concept in payload:
            for edge in self.edges:
                # each edge contains fields "sub", "pred", "obj"
                sub = edge["sub"]
                obj = edge["obj"]
                predicate = edge["pred"]

                # Edge example: {"sub": "Sensor", "pred": "is_a", "obj": "ElectronicDevice"}
                if query_concept == sub:
                    element = {
                        "query_concept": query_concept,
                        "concept": obj,
                        "relationship": predicate,
                    }
                    result.append(element)
                elif query_concept == obj:
                    element = {
                        "query_concept": query_concept,
                        "concept": sub,
                        "relationship": predicate,
                    }
                    result.append(element)

        return result

    def get_ontology_json(self):
        """
        Method used for fetching COGLO ontology as JSON
        :return: ontology (JSON)
        """
        # Ontology wasn't parsed correctly, just return empty array
        if self.ontology is None:
            return {}
        return self.ontology
