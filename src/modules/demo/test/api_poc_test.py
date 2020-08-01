import os
import pickle
import random

from src.modules.demo.api_poc import Parcel
from src.modules.demo.api_poc import Vehicle
from src.modules.demo.api_poc import VrpProcessor
from src.modules.partitioning.post_partitioning import GraphPartitioner

JSON_GRAPH_DATA_PATH = 'modules/demo/data/Graph_PoC.json'
MSB_FWD = 'http://116.203.13.198/api/postRecommendation'

##############################
pickle_path = './modules/demo/data/graphs.pickle'
#pickle_path = './' + JSON_GRAPH_DATA_PATH.replace('/', '_') + '.graphs.pickle'
partitioner = None
print('Checking if data from', JSON_GRAPH_DATA_PATH, 'exists.')
if os.path.exists(pickle_path):
    with open(pickle_path, 'rb') as loadfile:
        partitioner = pickle.load(loadfile)
    print('Loaded pickled graph data')

else:
    # make 25 partitions, so our VRP can do the work in reasonable time,
    # even then small or even-sized partitions are not guaranteed
    K = 1
    # instance partitioner object, partition input graph, create graph processors
    # for all partitions and then create instance of vrp proc
    print('No data found, runing load and partition procedure')
    partitioner = GraphPartitioner(JSON_GRAPH_DATA_PATH)
    partitioner.partition(K)
    with open(pickle_path, 'wb') as dumpfile:
        pickle.dump(partitioner, dumpfile)
        print('Stored pickled dump for future use')

vrpProcessor = VrpProcessor(partitioner.graphProcessors)

if __name__ == '__main__':

    # this is an example code that demoes input and computation of routing
    min_graph = partitioner.graphProcessors[0]
    for g in partitioner.graphProcessors:
        if len(g.nodes) < len(min_graph.nodes):
            min_graph = g

    available_vehicles = []
    dispatch_node = random.choice(min_graph.nodes).id
    for i in range(3):
        available_vehicles.append(Vehicle('Vehicle' + str(i), start_node=dispatch_node))

    requested_deliveries = []
    for i in range(15):
        requested_deliveries.append(
            Parcel(random.randint(100, 300), random.choice(min_graph.nodes).id, random.randint(1, 30)))

    print(['Vehicle {} at {}'.format(x.name, x.start_node) for x in available_vehicles])
    print(['Delivery of {} to {}'.format(x.volume, x.target) for x in requested_deliveries])

    vrpProcessor.process(available_vehicles, requested_deliveries, "SLO-CRO")
