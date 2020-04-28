import time
from math import inf

from ...vrp.vrp import VRP
from ...utils.structures.parcel import Parcel
from ...utils.structures.deliveries import Deliveries
from ...utils.structures.plan import Plan
from ...utils.structures.vehicle import Vehicle
from ...create_graph.config.config_parser import ConfigParser

config_parser = ConfigParser()
import numpy as np


class VrpProcessor:
    """Processes a request for routing
        Many operations on lists in this code can be replaced with dicts or similar, to remove list iteration with
        dict lookup.
    """
    def __init__(self, graphs, use_case):
        self.vrp = VRP()
        self.graphs = graphs
        self.use_case = use_case

    def map_vehicles(self, vehicles):
        """Assign vehicles to partitions"""
        map_v = [[] for _ in self.graphs]

        for v in vehicles:
            for i in range(len(self.graphs)):
                graph = self.graphs[i]
                for node in graph.nodes:
                    if v.start_node == node.id:
                        map_v[i].append(v)

        return map_v

    def map_deliveries(self, deliveries):
        """Assign deliveries to partitions"""
        delivery_parts = [[] for _ in self.graphs]
        for d in deliveries:
            for i in range(len(self.graphs)):
                graph = self.graphs[i]
                nodes = graph.nodes
                for n in nodes:
                    if n.id == d.target:
                        delivery_parts[i].append(d)
        print(sum([len(x) for x in delivery_parts]), len(deliveries))
        assert sum([len(x) for x in delivery_parts]) == len(deliveries)
        return delivery_parts

    @staticmethod
    def map_dropoff(graph, deliveries):
        """Computer VRP input vector, how much volume will be dropped off on each node"""
        dropoff = [0] * len(graph.nodes)
        for d in deliveries:
            node = graph.node_from_id(d.target)
            idx = graph.nodes.index(node)
            dropoff[idx] += d.volume
        return dropoff

    @staticmethod
    def map_start_nodes(graph, vehicles):
        """Compute VRP input list of vectors where do the vehicles start"""
        indexes = []
        for v in vehicles:
            index = []
            for i in range(len(graph.nodes)):
                if graph.nodes[i].id == v.start_node:
                    index.append(i)
                for parcel in v.parcels:
                    if graph.nodes[i].id == parcel.target:
                        index.append(i)
            indexes.append(index)
        return indexes

    def process(self, vehicles, deliveries_all):
        """Process routing request with N vehicles and M deliveries, to produce a list of routing plans"""
        deliveries = deliveries_all.deliveries
        deliveries_req = deliveries_all.req

        # Handle SLO-CRO use case for mapping vehicles and deliveries
        if self.use_case == "SLO-CRO":
            delivery_map = self.map_slo_cro_deliveries(deliveries)
            delivery_map_req = self.map_slo_cro_deliveries(deliveries_req)
            vehicle_map = self.map_slo_cro_vehicles(vehicles)
        else:
            delivery_map = self.map_deliveries(deliveries)
            delivery_map_req = self.map_deliveries(deliveries_req)
            vehicle_map = self.map_vehicles(vehicles)

        # mapping all data to partitions
        plans = [Plan(vehicle_map[i], delivery_map[i], delivery_map_req[i], self.graphs[i]) for i in
                 range(len(self.graphs))]
        routes = []

        for i, plan in enumerate(plans):
            partition = plan.partition
            if len(plan.deliveries) == 0 or len(plan.vehicles) == 0:
                print('Skipping plan {}, no deliveries or vehicles assigned'.format(i))
                continue
            else:
                print("Computing routing for plan", i)
            print('Starting planning for {} vehicles to deliver {} packages. Node len: {}'.format(len(plan.vehicles),
                                                                                                  len(plan.deliveries),
                                                                                                  len(partition.nodes)))
            # compute input vectors
            dropoff = self.map_dropoff(plan.partition, plan.deliveries)
            capacity = [v.capacity for v in plan.vehicles]
            start_loc = self.map_start_nodes(partition, plan.vehicles)
            costs = [e.cost for e in partition.edges]

            # ============================================================================================
            # MODIFICATION FOR STOPAR'S VRP
            # ============================================================================================

            # Only one vehicle per plan
            indexes = []
            for v in plan.vehicles:
                for i in range(len(partition.nodes)):
                    if partition.nodes[i].id == v.start_node:
                        indexes.append(i)
            start_loc_stopar = indexes

            # Cost matrix per kilometer
            cost_matrix = []
            for vehicle in plan.vehicles:
                for edge in partition.edges:
                    cost_matrix.append(edge.cost/1000) # to kms
            new_cost_matrix = cost_matrix #+ cost_matrix

            """Build incidence matrix for LS VRP instance"""
            incident_matrix = []
            graph = plan.partition

            for ni, n in enumerate(graph.nodes):
                tmp_arr = [0] * len(graph.edges)
                for ne, e in enumerate(graph.edges):
                    if e.start == n.id:
                        tmp_arr[ne] = -1
                    elif e.end == n.id:
                        tmp_arr[ne] = 1
                tmp_arr_neg = np.negative(tmp_arr).tolist()
                incident_matrix.append(tmp_arr)

            req_json = {}
            req_json['startV'] = start_loc_stopar
            req_json['endV'] = start_loc_stopar
            req_json['vehicleCapacityV'] = capacity
            req_json['nodeDistributionV'] = dropoff

            edge_vector = [edge.cost/(60*1000) for edge in partition.edges]
            req_json['edgeTimeV'] = edge_vector# + edge_vector
            req_json['nodeOpenV']= [0 for _ in partition.nodes] # from 0
            req_json['nodeCloseV']= [16 for _ in partition.nodes] # to 16
            req_json['costMat'] = new_cost_matrix
            req_json["incidenceMat"] = incident_matrix

            computed_routes, costs = self.vrp.vrp(req_json)
            dispatch = dropoff

            # ====================================================================================
            # END
            # ====================================================================================

            # TODO: uncomment to use MIHA's VRP
            # computed_routes, dispatch, objc = self.vrp.vrp(partition.incident_matrix, dropoff, capacity, start_loc, costs)

            # compute routes based on dispatch vectors from VRP. Since VRP output is incomplete/not best,
            # we add A* routing on top
            plan_routes = self.make_route(computed_routes, dispatch, partition,
                                          plan.vehicles, plan.deliveries, plan.deliveries_req)
            routes += plan_routes

        return routes

    @staticmethod
    def find_closest_post(loads, start, graph):
        nodes = graph.nodes
        min_dist = inf
        min_idx = -1

        for post_idx, load in enumerate(loads):
            if load > 0:
                cur_dist = graph.distance(start, nodes[post_idx])
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_idx = post_idx
        return min_idx

    def make_route(self, graph_routes, loads, graph, vehicles, deliveries, deliveries_req):
        nodes = graph.nodes
        edges = graph.edges
        print("Building route from VRP output...")
        print(len(edges))
        start_nodes = [graph.node_from_id(x.start_node) for x in vehicles]
        start_time = time.time()
        routes = []
        converted_routes = []
        loads_new = []
        vehicle_node_sequence = []

        # add new parcels to vehicle.parcels list
        for x, vehicle in enumerate(vehicles):
            load = loads[x]
            loads_origin = self.map_dropoff(graph, vehicle.parcels)
            for i in range(len(nodes)):
                vehicle_load_diff = load[i] - loads_origin[i]
                while vehicle_load_diff > 0:
                    for j in range(len(deliveries_req)):
                        if deliveries_req[j].target == nodes[i].id:
                            vehicle.parcels.append(deliveries_req[j])
                            deliveries_req.remove(deliveries_req[j])
                            vehicle_load_diff -= 1
                            break
            loads_new.append(self.map_dropoff(graph, vehicle.parcels))

        for i, vehicle_load in enumerate(loads_new):
            start_node = start_nodes[i]
            route = [start_node]
            vehicle_node_sequence.append(start_node.id)

            current_node = start_node
            vehicle_load[nodes.index(current_node)] -= vehicle_load[nodes.index(current_node)]
            cost_astar = 0
            vehicle_load = list(map(int, vehicle_load))
            dispatch = vehicle_load.copy()
            # variableIDS_list = []
            # always find closest post with parcels to pick/drop off.
            # start at closest node
            # get route and clear up any packages on this route

            while sum(vehicle_load) > 0:  # run until all parcels have been delivered
                post_idx = self.find_closest_post(vehicle_load, current_node,
                                                  graph)  # idx of closest post with parcel demand
                target = nodes[post_idx]  # convert idx to node object
                vehicle_load[post_idx] -= vehicle_load[post_idx]  # take/drop all parcels
                route.append(target)


                #partial_path = graph.get_path(current_node, target)  # get path from current to next dropoff node
                #for node in partial_path.path:  # drop off parcels along the way to target
                #   for idx, val in enumerate(vehicle_load):
                #      if val > 0 and nodes.index(node) == idx:
                #          vehicle_load[idx] -= vehicle_load[idx]

                #current_node = target  # we are now at new node
                #vehicle_node_sequence.append(current_node.id)
                #cost_astar += partial_path.cost
                # merge existing and new route, avoid adding duplicate node on start of route
                #route += partial_path.path if len(partial_path.path) == 1 else partial_path.path[1:]

            # debug info
            #print("Vehicle: {}".format(vehicles[i].name), "|", "parcels:", str(vehicles[i].parcels))
            #print("Edges: VRP: {}, A*:{}".format(sum(graph_routes[i]), len(route)))
            # calculate theoretical cost of all visited edges in vrp, to compare to A*
            #cost_vrp = sum([count * edges[j].cost for j, count in enumerate(graph_routes[i])])
            #print("Cost VRP: {}, Cost A*:{}".format(cost_vrp, cost_astar))
            # print([item if x > 0 else -1 for item, x in enumerate(original_vehicle_load)])
            graph.print_path(route)
            routes.append(route)
            converted_routes.append({"UUID": vehicles[i].name, "route": self.map_parcels_to_route(route, dispatch, graph, vehicles[i])})


        #print("Route build took: {}s".format(time.time() - start_time))
        print("calculated routes:", routes)
        print("converted routes:", converted_routes)

        return converted_routes

    @staticmethod
    def map_parcels_to_route(route, loads, graph, vehicle):
        """Maps parcels UUIDs to the vehicle route: existing parcels on hte vehicles + additional parcels alocated
        loads_diff: position of addtional parcels to the vehicles routes from adhoc order"""
        nodes = graph.nodes
        converted_route = []
        parcel_list = vehicle.parcels

        # map parcel UUIDs to route
        if len(route) == 1 and loads[nodes.index(route[0])] == 0:
            return []

        for idx, node in enumerate(route):
            node_idx = nodes.index(node)

            parcels = [x.uuid for x in parcel_list if x.target == node.id]

            if (int(loads[node_idx]) > 0 or idx == 0):
                converted_route.append({
                    "locationId": node.id,
                    "dropoffWeightKg": int(loads[node_idx]),
                    # "dropoffVolumeM3": int(loads[node_idx] / 10),
                    "parcels": parcels,
                    "info": "This parcel must be delivered to location " + str(node.id),
                    "position": "{},{}".format(node.lon, node.lat)
                })
                for parcel_remove in parcel_list:  # removes the added parcels from the pending parcel list
                    if parcel_remove.uuid in parcels:
                        parcel_list.remove(parcel_remove)

        return converted_route

    @staticmethod
    def parse_vehicles(clos):
        """Create a list of Vehicle objects from JSON input"""
        vehicles = []
        for clo in clos:
            parcels = []
            for parcel in clo["parcels"]:
                parcels.append(Parcel(parcel["UUIDParcel"], parcel["destination"],
                                      parcel["weight"], clo["currentLocation"]))
                # parcels.append(parcel["UUIDParcel"])
            capacity = clo["capacity"] - len(parcels)
            vehicles.append(Vehicle(clo["UUID"], clo["currentLocation"], parcels, capacity))
        return vehicles

    @staticmethod
    def parse_deliveries(clos, requests):
        """Create a list of Vehicle objects from JSON input"""
        deliveries_origin = []
        # list of additional parcels from request
        deliveries_diff = [Parcel(x["UUIDParcel"], x["destination"],
                                  x["weight"], x["pickup"]) for x in requests]
        # list of parcels on CLOs before request
        for clo in clos:
            for parcel in clo["parcels"]:
                deliveries_origin.append(Parcel(parcel["UUIDParcel"], parcel["destination"],
                                                parcel["weight"], clo["currentLocation"]))
        deliveries_all = deliveries_origin + deliveries_diff
        deliveries = Deliveries(deliveries_origin, deliveries_diff, deliveries_all)
        return deliveries

    ####################################################################################
    # Helper methods and methods used for specific use case or purpose
    ####################################################################################

    def map_slo_cro_vehicles(self, vehicles):
        """
        Map vehicles to first or second graph. First graph represents SLO nodes and the second
        graphs represents CRO nodes.
        :param vehicles:
        :return:
        """
        map_v = [[] for _ in self.graphs]

        # First graph is SLO, second graph is CRO
        for v in vehicles:
            if "S" in v.start_node:
                map_v[0].append(v)
            elif "H" in v.start_node:
                map_v[1].append(v)
            else:
                print("Vehicle start node does not have 'S' or 'H'.")
                exit(1)

        return map_v

    def map_slo_cro_deliveries(self, deliveries):
        """
        Map deliveries for SLO-CRO use case. For each parcel, we check current (start) location
        and destination (target). If parcel needs to be delivered from SLO to CRO, we will
        assign the closest border node as target, if CRO node assigned that is not the border node.
        :param deliveries:
        :return:
        """
        delivery_parts = [[] for _ in self.graphs]

        for parcel in deliveries:
            if "S" in parcel.current_location:
                if "H" in parcel.target:
                    # assign closest cro border node
                    cro_border_nodes = config_parser.get_border_nodes_cro()
                    if parcel.target not in cro_border_nodes:
                        # TODO: assign the closest node instead of the first one
                        parcel.target = cro_border_nodes[0]
                delivery_parts[0].append(parcel)
            elif "H" in parcel.current_location:
                if "S" in parcel.target:
                    # assign closest slo border node
                    slo_border_nodes = config_parser.get_border_nodes_slo()
                    if parcel.target not in slo_border_nodes:
                        # TODO: assign the closest node instead of the first one
                        parcel.target = slo_border_nodes[0]
                delivery_parts[1].append(parcel)
            else:
                print("Current parcel location is not 'S' nor 'H'!")
                exit(1)

        return delivery_parts

