from src.modules.create_graph.create_graph import JsonGraphCreator

creator = JsonGraphCreator()
use_case = "SLO-CRO_crossborder"
creator.create_json_graph(use_case)
