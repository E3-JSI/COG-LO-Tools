import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
import time


class VRP:
    def __init__(self):
        pass

    @staticmethod
    def vrp(graph_incidence_mat, demand, capacity_vec, start_loc_vec, edges_length):
        start_time = time.time()
        # Data validity
        if len(start_loc_vec) != len(capacity_vec):
            raise ValueError('Number of vehicles & number of start locations not match!')
        if sum(capacity_vec) < sum(demand):
            raise ValueError('Total vehicles capacity to low!')
        if len(graph_incidence_mat) != len(demand):
            raise ValueError('Number of nodes in dispatch and incidence matrix dont match!')
        if len(edges_length) != len(graph_incidence_mat[0]):
            raise ValueError('Size of edges_length and n_edges do not match!')
        E = graph_incidence_mat

        # Additional Variables
        n_cycles = np.size(capacity_vec)
        n_edges = len(E[1])
        n_nodes = len(E)

        offset_c = 0
        offset_k = n_edges * n_cycles
        offset_o = n_cycles * n_edges + n_cycles * n_nodes
        # number of columns in matrix A1+A2+A3, variables X, K, O
        offset_a = 2 * n_nodes * n_cycles + n_edges * n_cycles

        # Full A MATRIX
        # CONSTRAINT IV - UPDATED - there is od number of n_edges on cycles
        # the right side of the equation should be zeros(n_edges*n_cycles + n_vert*n_cycles)
        # b variables = size([capacity_vec])= n_cycles*n_edges + n_cycles*n_nodes

        AddRow = sum(len(mandatory_location_list) for mandatory_location_list in start_loc_vec)
        rows_A1 = 2 * n_nodes * n_cycles + n_nodes * n_cycles + n_edges * n_cycles + AddRow
        cols_A1 = n_nodes * n_cycles + n_edges * n_cycles

        A1 = np.zeros((rows_A1, cols_A1))
        b1 = ([0] * (rows_A1 - AddRow)) + ([-1] * AddRow)

        offset_block0 = 0
        offset_block1 = n_nodes * n_cycles
        offset_block2 = 2 * n_nodes * n_cycles
        offset_block3 = 3 * n_nodes * n_cycles
        offset_block4 = 3 * n_nodes * n_cycles + n_cycles * n_edges

        for k in range(n_cycles):
            for i in range(n_nodes):
                for j in range(n_edges):
                    # C_kj * E_ij - 2K_ki == 0
                    A1[offset_block0 + k * n_nodes + i, offset_c + k * n_edges + j] = E[i][j]
                    A1[offset_block1 + k * n_nodes + i, offset_c + k * n_edges + j] = -E[i][j]
                    A1[offset_block0 + k * n_nodes + i, offset_k + k * n_nodes + i] = -2
                    A1[offset_block1 + k * n_nodes + i, offset_k + k * n_nodes + i] = 2
                    # K_ki >= 0
                    A1[offset_block2 + k * n_nodes + i, offset_k + k * n_nodes + i] = -1

        for k in range(0, n_cycles):
            for j in range(0, n_edges):
                A1[offset_block3 + k * n_edges + j, offset_c + k * n_edges + j] = -1

        insert_line = 0
        for k in range(n_cycles):
            mandatory_location_list = start_loc_vec[k]
            for mandatory_node_id in mandatory_location_list:
                for j in range(n_edges):
                    A1[offset_block4 + insert_line, offset_c + k * n_edges + j] = -E[mandatory_node_id][j]
                insert_line = insert_line + 1

        # CONSTRAINT II - A2 - the number of packets delivered on the node is equal to all total demand on the node
        # number of rows: 2 * num. of n_nodes
        # number of columns: num. n_nodes * num. of cycles
        # variables Oki
        # B vector = [dispatch_vec, -dispatch_vec]
        b2 = demand + [-val for val in demand] + ([0] *(n_cycles * n_nodes))

        rows_A2 = 2 * n_nodes + n_nodes * n_cycles
        cols_A2 = n_nodes * n_cycles
        A2 = np.zeros((rows_A2, cols_A2))
        for i in range(0, n_nodes):
            for k in range(0, n_cycles):
                A2[i, n_nodes * k + i] = 1
                A2[i + n_nodes, n_nodes * k + i] = -1
                A2[i * n_cycles + 2 * n_nodes + k, i * n_cycles + k] = -1

        # CONSTRAINT III A3 matrix: sum of load on each vehicles must not exceed vehicle load capacity
        # variables Oki
        rows_A3 = n_cycles
        cols_A3 = n_cycles * n_nodes
        b3 = capacity_vec.copy()
        b23 = b2 + b3

        A3 = np.zeros((rows_A3, cols_A3))
        for k in range(0, n_cycles):  # for each vehicle
            for i in range(0, n_nodes):  # take the sum of load on enach node (sum on Ow variables)
                A3[k, i + k * n_nodes] = 1
        A23 = np.vstack([A2, A3])

        # CONSTRAINT I - total number of all packets delivered is equal to summ of all dispatch_vec
        n_vars = 2 * n_nodes * n_cycles + n_edges * n_cycles  # number of columns in matrix A1+A2+A3, variables X, K, O
        n_slacks = n_cycles * n_edges * n_nodes

        cols_A4 = n_vars + n_slacks

        # constraint 4.1. SUM for ijk -> (Eij * Aijk)
        A41 = np.zeros((n_nodes * n_cycles, n_vars + n_slacks))
        for k in range(n_cycles):
            for i in range(n_nodes):
                rowN = i * n_cycles + k
                A41[rowN, offset_o + k * n_nodes + i] = 2
                for j in range(n_edges):
                    A41[rowN, offset_a + i * n_edges * n_cycles + j * n_cycles + k] = -E[i][j]

        # Adding constraints 4.2.: Aijk - Cki*di <= 0

        A42 = np.zeros((n_slacks, cols_A4))
        for i in range(0, n_nodes):
            for j in range(0, n_edges):
                for k in range(0, n_cycles):
                    A42[i * n_edges * n_cycles + j * n_cycles + k, n_edges * k + j] = -demand[i]
                    A42[i * n_edges * n_cycles + j * n_cycles + k,
                        2 * n_cycles * n_nodes + n_cycles * n_edges + i * n_edges * n_cycles + j * n_cycles + k] = 1

        # constraint 4.3.: Aijk - Oki <= 0
        A43 = np.zeros((n_slacks, cols_A4))

        offset = 0
        offset1 = n_edges * n_cycles + n_nodes * n_cycles
        offset2 = n_edges * n_cycles + 2 * n_nodes * n_cycles
        for i in range(0, n_nodes):
            for j in range(0, n_edges):
                for k in range(0, n_cycles):
                    # print (offset+i*n_edges*n_cycles+j*n_cycles+k, offset1 + n_nodes*k+i)
                    A43[offset + i * n_edges * n_cycles + j * n_cycles + k, offset1 + n_nodes * k + i] = -1
                    A43[
                        offset + i * n_edges * n_cycles + j * n_cycles + k,
                        offset2 + i * n_edges * n_cycles + j * n_cycles + k] = 1

        A4 = np.vstack([A41.copy(), A42[0:n_slacks, :]])
        A4 = np.vstack([A4, A43])

        b4 = [0] * ((n_nodes * n_cycles) + (2 * n_cycles * n_nodes * n_edges))

        # FINAL MATRIX  - A with all constraints
        # concatenate A1 and A23 = A matrix
        A1extend = np.c_[A1, np.zeros((len(A1), len(A23[0])))]
        A23extend = np.c_[np.zeros((len(A23), len(A1[0]))), A23]



        A123 = np.vstack([A1extend.copy(), A23extend[0:len(A23)]])

        # Final concate A123 & A4
        A123extend = np.c_[A123, np.zeros((len(A123), n_nodes * n_edges * n_cycles))]
        timea = time.time()
        A = np.vstack([A123extend.copy(), A4])
        endtime = time.time() - timea
        print("Vstack took: {} for {}".format(endtime, A.shape))

        b = b1 + b23 + b4

        # CREATE VARIABLES  X - vector with c11-cnn variables

        # capacity_vec variables
        C = ['C_' + str(k) + '_' + str(j) for k in range(n_cycles) for j in range(n_edges)]
        # K variables
        K = ['K_' + str(k) + '_' + str(i) for k in range(n_cycles) for i in range(n_nodes)]
        # \"Ow\" variables
        Ow = ['O_' + str(k) + '_' + str(i) for k in range(n_cycles) for i in range(n_nodes)]

        # \"Aijk\" variables
        Aijk = []
        for i in range(n_nodes):
            for j in range(n_edges):  # numer of vehicles
                for k in range(n_cycles):
                    Aijk.append('A_' + str(i) + '_' + str(j) + '_' + str(k))

        # Declaring the solver
        solver = pywraplp.Solver('SolveIntegerProblem',
                                 pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        x_min = 0.0  # lower variables border
        x_max = solver.infinity()  # Upper variables border

        variables = []
        for varN, xi_name in enumerate(C):  # declaring objective variables [C1.... Cn]
            variables.append(solver.IntVar(x_min, x_max, xi_name))
        for varN, xi_name in enumerate(K):  # declaring slack variables
            variables.append(solver.IntVar(x_min, x_max, xi_name))
        for varN, xi_name in enumerate(Ow):  # declaring load dispatch variables
            variables.append(solver.NumVar(x_min, x_max, xi_name))
        for varN, xi_name in enumerate(Aijk):  # declaring load dispatch variables
            variables.append(solver.NumVar(x_min, x_max, xi_name))

        # DECLARE CONSTRAINTS
        for rowN, row in enumerate(A):
            left_side = None
            for colN, coeff in enumerate(row):
                if coeff == 0:
                    continue
                elif left_side is None:
                    left_side = coeff * variables[colN]
                else:
                    left_side += coeff * variables[colN]
            if left_side is None and b[rowN] < 0:
                # if left_side is None and b[0,rowN] < 0:
                raise ValueError('Constraint ' + str(rowN) + ' cannot be satisfied!')
            if left_side is not None:
                # solver.Add(left_side <= t[0,rowN])
                solver.Add(left_side <= b[rowN])

        # DECLARE OBJECTIVE FUNCTION & INVOKE THE SOLVER
        cost = 0
        coeffs = edges_length
        for k in range(n_cycles):
            for coeffN in range(n_edges):
                cost += variables[offset_c + k * n_edges + coeffN] * coeffs[offset_c + coeffN]

        solvetime = time.time()
        # run the optimization and check if we got an optimal solution
        solver.Minimize(cost)
        result_status = solver.Solve()
        assert result_status == pywraplp.Solver.OPTIMAL
        endsolve = time.time() - solvetime
        print("Solver took: {}".format(endsolve))

        obj_val = solver.Objective().Value()

        routes = []
        Omatrix = []

        for k in range(n_cycles):
            # edges
            C_row = []
            for j in range(n_edges):
                idx1 = offset_c + n_edges * k + j
                val1 = int(variables[idx1].solution_value())
                C_row.append(val1)
            routes.append(C_row)

            # nodes
            O_row = []
            for i in range(n_nodes):
                O_row.append(variables[offset_o + n_nodes * k + i].solution_value())
            Omatrix.append(O_row)

        # return function - results
        print("VRP total execution took: {}".format(time.time() - start_time))
        return routes, Omatrix, obj_val
