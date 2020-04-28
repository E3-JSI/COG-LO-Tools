function [E_result, C_result, T_result, T_clique_orig] = get_clique(E, C, edge_time_vec)

    n_vehicles = size(C, 1);
    n_nodes = size(E, 1);
    n_edges = size(E, 2);
    ones_e = ones(n_edges, 1);
    A_cost = zeros(n_nodes, n_nodes, n_vehicles);
    A_time = zeros(n_nodes, n_nodes, n_vehicles);
    
    E_in = max(E, 0);
    E_out = max(-E, 0);
    J = 1:n_edges;
    
    M_node_edge = E_out * sparse(J, J, J) * E_in';
    
    function [edgeN] = nodeN2edgeN(srcN, dstN, n_nodes)
        offset_dst = dstN;
        if dstN > srcN
            offset_dst = dstN-1;
        end

        edgeN = (srcN-1)*(n_nodes-1) + offset_dst;
    end
    
    T_clique_orig = zeros(n_nodes*(n_nodes-1), n_edges);
    
    disp('constructing adjacency matrices');
    for vehicleN = 1:n_vehicles
        c_vehicle = C(vehicleN, :);
        
        A_cost_vehicle = incidence2adj_mat(E, c_vehicle);
        A_time_vehicle = incidence2adj_mat(E, edge_time_vec);
        
        A_cost(:, :, vehicleN) = A_cost_vehicle;
        A_time(:, :, vehicleN) = A_time_vehicle;
    end

    
    disp('constructing cliques');
    for vehicleN = 1:n_vehicles
        A_cost_vehicle = A_cost(:, :, vehicleN);
        %A_time_vehicle = A_time(:, :, vehicleN);
        
        A_cost_clique = zeros(n_nodes, n_nodes);
        A_time_clique = zeros(n_nodes, n_nodes);
        for srcN = 1:n_nodes
            [neighbour_cost_vec, P_edge] = find_neighbour_wgt_vec(srcN, A_cost_vehicle, M_node_edge);
%             [neighbour_time_vec, ~] = find_neighbour_wgt_vec(srcN, A_time_vehicle, M_node_edge);
            
            neighbour_time_vec = P_edge*ones_e;
            
            A_cost_clique(srcN, :) = neighbour_cost_vec;
            A_time_clique(srcN, :) = neighbour_time_vec;
            
            for dstN = 1:n_nodes
                if srcN ~= dstN
                    clique_edgeN = nodeN2edgeN(srcN, dstN, n_nodes);
                    T_clique_orig(clique_edgeN, :) = P_edge(dstN, :);
                end
            end
        end
        
        A_cost(:, :, vehicleN) = A_cost_clique;
        A_time(:, :, vehicleN) = A_time_clique;
    end
    
    n_output_edges = n_nodes*(n_nodes-1);
    
    % construct the output
    E_result = zeros(n_nodes, n_output_edges);
    C_result = zeros(n_vehicles, n_output_edges);
    T_result = zeros(n_vehicles, n_output_edges);
    
    for srcN = 1:n_nodes
        for dstN = 1:n_nodes
            if srcN == dstN
                continue;
            end
            
            edgeN = nodeN2edgeN(srcN, dstN, n_nodes);
            
            E_result(srcN, edgeN) = -1;
            E_result(dstN, edgeN) = 1;
        end
    end
    
    for vehicleN = 1:n_vehicles        
        for srcN = 1:n_nodes
            for dstN = 1:n_nodes
                if srcN == dstN
                    continue;
                end
                
                edge_cost = A_cost(srcN, dstN, vehicleN);
                edge_time = A_time(srcN, dstN, vehicleN);

                edgeN = nodeN2edgeN(srcN, dstN, n_nodes);
                
                C_result(vehicleN, edgeN) = edge_cost;
                T_result(vehicleN, edgeN) = edge_time;
            end
        end
    end
end