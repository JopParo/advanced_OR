import math
import numpy as np
import pandas as pd
import random
import copy

'Reads the input file and stores every relevant information in a dictionary'
def Read_gvrp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    nodes = {}      # Stores coordinates of nodes
    clusters = {}   # Stores cluster -> list of nodes
    demands = {}    # Stores demand per cluster
    depot = None
    vehicle_capacity = None
    num_vehicles = None
    
    mode = None
    cluster_id = 1  # Clusters start from 1

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == 'DIMENSION':
            num_nodes = int(parts[-1])
        elif parts[0] == 'VEHICLES':
            num_vehicles = int(parts[-1])
        elif parts[0] == 'CAPACITY':
            vehicle_capacity = int(parts[-1])
        elif parts[0] == 'NODE_COORD_SECTION':
            mode = 'NODES'
        elif parts[0] == 'GVRP_SET_SECTION':
            mode = 'CLUSTERS'
            cluster_id = 1  # Reset for clusters
        elif parts[0] == 'DEMAND_SECTION':
            mode = 'DEMANDS'
            cluster_id = 1  # Reset for demand assignment
        elif parts[0] == 'EOF':
            break
        elif mode == 'NODES':
            node_id, x, y = map(int, parts)
            nodes[node_id] = (x, y)

        elif mode == 'CLUSTERS':
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            cluster_id
            for value in parts[:-1]:  # Ignore the "-1" at the end
                clusters[cluster_id].append(int(value))
            clusters[cluster_id].pop(0)
            cluster_id += 1
        elif mode == 'DEMANDS':
            demands[cluster_id] = int(parts[1])  # Assign demand to the cluster
            cluster_id += 1  # Move to next cluster
    distance_matrix = Compute_distance_matrix(nodes)
    return nodes, clusters, demands, vehicle_capacity, num_vehicles, distance_matrix

'Computes distances between given nodes according to the Euclidean formula'
def Compute_distance_matrix(nodes):
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes + 1, num_nodes + 1))
    for i in nodes:
        for j in nodes:
            if i != j:
                x1, y1 = nodes[i]
                x2, y2 = nodes[j]
                distance_matrix[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance_matrix

'Constructs a route inside each cluster starting at a random customer and using nearest neighbor'
def Intra_cluster_route(cluster_nodes, distance_matrix):
    if not cluster_nodes:
        return []
    start_node = random.choice(cluster_nodes)
    route = [start_node]
    unvisited = set(cluster_nodes)
    unvisited.remove(start_node)
    while unvisited:
        last_node = route[-1]
        nearest_node = min(unvisited, key=lambda node: distance_matrix[last_node][node])
        route.append(nearest_node)
        unvisited.remove(nearest_node)
    return route

'Creates an initial solution: adds closest cluster (centre) to current vehicle pos. respecting the capacity'
def Initial_solution(nodes, clusters, demands, vehicle_capacity, num_vehicles, distance_matrix):
    depot = 1
    unvisited_clusters = set(clusters.keys())
    vehicle_routes = [[] for _ in range(num_vehicles)]
    for vehicle in range(num_vehicles):
        current_location = depot
        remaining_capacity = vehicle_capacity
        vehicle_clusters = [depot]
        while unvisited_clusters:
            possible_clusters = [
                c for c in unvisited_clusters if demands[c] <= remaining_capacity
            ]
            if not possible_clusters:
                break
            next_cluster = min(
                possible_clusters, key=lambda c: distance_matrix[current_location][clusters[c][0]]
            )
            vehicle_clusters.append(next_cluster)
            remaining_capacity -= demands[next_cluster]
            current_location = clusters[next_cluster][-1]
            unvisited_clusters.remove(next_cluster)
        vehicle_clusters.append(depot)
        vehicle_routes[vehicle] = vehicle_clusters
    return vehicle_routes

'Creates an random initial solution'
def Random_initial_solution(nodes, clusters, demands, vehicle_capacity, num_vehicles, distance_matrix):
    depot = 1

    while True:
        vehicle_routes = [[] for _ in range(num_vehicles)]
        vehicle_loads = [0 for _ in range(num_vehicles)]
        unvisited_clusters = list(clusters.keys())
        random.shuffle(unvisited_clusters)
        success = True

        for cluster in unvisited_clusters:
            assigned = False
            vehicle_indices = list(range(num_vehicles))
            random.shuffle(vehicle_indices)
            for k in vehicle_indices:
                if vehicle_loads[k] + demands[cluster] <= vehicle_capacity:
                    if not vehicle_routes[k]:
                        vehicle_routes[k].append(depot)
                    vehicle_routes[k].append(cluster)
                    vehicle_loads[k] += demands[cluster]
                    assigned = True
                    break
            if not assigned:
                success = False
                break
        if success:
            for k in range(num_vehicles):
                if vehicle_routes[k]:
                    vehicle_routes[k].append(depot)
                else:
                    vehicle_routes[k] = [depot, depot]
            return vehicle_routes

'Prints a given solution'
def Print_solution(solution, clusters):
    for v, route in enumerate(solution):
        formatted_route = [1]
        for cluster in route[1:-1]:
            if cluster in clusters:
                formatted_route.append(clusters[cluster])
            else:
                formatted_route.append(cluster)
        formatted_route.append(1)
        print(f"Vehicle {v+1} Route: {formatted_route}")

'Tweak Operator: Intra-route (cluster level) using heuristics "Relocate" and "Exchange"'
def Intra_route(current_solution, clusters):
    K = len(current_solution)
    k = random.choice(range(K))
    route = current_solution[k]
    random_cluster = random.choice(route[1:-1])
    cluster = clusters[random_cluster]
    
    if len(cluster) > 1:
        heuristic = random.choice(["Relocate", "Exchange"])
    else:
        heuristic = "Relocate"
    if heuristic == "Relocate":
        random_customer = random.choice(cluster)
        cluster.remove(random_customer)
        random_index = random.randint(0, len(cluster))
        cluster.insert(random_index, random_customer)
        clusters[random_cluster] = cluster
    if heuristic == "Exchange":
        customer1, customer2 = random.sample(range(len(cluster)), 2)
        cluster[customer1], cluster[customer2] = cluster[customer2], cluster[customer1]
        clusters[random_cluster] = cluster
    return current_solution

'Tweak Operator: Inter-route (route level) using heuristics "Insert" and "Swap"'
def Inter_route(current_solution, clusters, demands, vehicle_capacity):
    K = len(current_solution)
    receiving_k = list(range(K))
    k = random.choice(range(K))
    receiving_k.remove(k)
    random_k = random.choice(receiving_k)
    route = current_solution[k]
    if len(route) > 2:
        random_cluster = random.choice(route[1:-1])
        heuristic = random.choice(["Insert", "Swap"])
        if heuristic == "Insert":
            if Route_demand(current_solution[random_k], clusters, demands) + demands[random_cluster] <= vehicle_capacity:
                route = route[1:-1]
                route.remove(random_cluster)
                route.insert(0, 1)
                route.append(1)
                random_index = random.randint(1, len(current_solution[random_k]) - 1)
                current_solution[random_k].insert(random_index, random_cluster)
                current_solution[k] = route
        if heuristic == "Swap":
            receiving_route = current_solution[random_k]
            if len(receiving_route) > 2:
                random_cluster_receiving = random.choice(receiving_route[1:-1])
                route = route[1:-1]
                receiving_route = receiving_route[1:-1]
                route.remove(random_cluster)
                route.insert(0,1)
                route.append(1)
                receiving_route.remove(random_cluster_receiving)
                receiving_route.insert(0,1)
                receiving_route.append(1)
                random_index = random.randint(1, len(receiving_route) - 1)
                receiving_route.insert(random_index, random_cluster)
                random_index = random.randint(1, len(route) - 1)
                route.insert(random_index, random_cluster_receiving)
                if Route_demand(route, clusters, demands) <= vehicle_capacity and Route_demand(receiving_route, clusters, demands) <= vehicle_capacity:
                    current_solution[k] = route
                    current_solution[random_k] = receiving_route
    return current_solution

'Tweak Operator: For a random route/vehicle changes the order of two random clusters'
def Cluster_reordering(current_solution, clusters, distance_matrix):
    K = len(current_solution)
    k = random.choice(range(K))
    route = current_solution[k]
    if len(route) <= 3:
        return current_solution
    index_random_cluster1, index_random_cluster2 = random.sample(range(1, len(route) - 1), 2)
    route[index_random_cluster1], route[index_random_cluster2] = route[index_random_cluster2], route[index_random_cluster1]
    current_solution [k] = route
    return current_solution

'Calculates distance for a given route'
def Calculate_route_distance(route, distance_matrix):
    expanded_route = [route[0]]
    for cluster_id in route[1:-1]:
        expanded_route.extend(clusters[cluster_id])
    expanded_route.append(route[-1])

    total_distance = 0
    for i in range(len(expanded_route) - 1):
        from_customer = expanded_route[i]
        to_customer = expanded_route[i + 1]
        total_distance += distance_matrix[from_customer][to_customer]
    return total_distance

'Calculates total distance for a given solution'
def Calculate_total_distance(current_solution, distance_matrix):
    total_distance = 0
    for route in current_solution:
        total_distance += Calculate_route_distance(route, distance_matrix)
    return total_distance

'Calculates the total demand for given route'
def Route_demand(route, clusters, demands):
    total_demand = 0
    route = route[1:-1]
    for cluster_id in route:
        total_demand += demands[cluster_id]
    return total_demand

'Randomly uses one of the three Tweak Operators for new solution'
def Tweak_solution(current_solution, clusters, demands, vehicle_capacity, distance_matrix, weights):
    #tweak_operation = random.choices(["intra", "inter", "reorder"], weights = weights, k = 1)[0]
    tweak_operation = random.choice(["intra", "inter", "reorder"])
    if tweak_operation == "intra":
        #print("Tweak Operator: intra")
        new_solution = Intra_route(current_solution, clusters)
    elif tweak_operation == "inter":
        #print("Tweak Operator: inter")
        new_solution = Inter_route(current_solution, clusters, demands, vehicle_capacity)
    else:
        #print("Tweak Operator: reorder")
        new_solution = Cluster_reordering(current_solution, clusters, distance_matrix)
    return new_solution

'Tabu Search implementation (FIFO for the list)'
def Tabu_search(initial_solution, clusters, demands, vehicle_capacity, distance_matrix, max_iterations, tabu_length, weights):
    l = tabu_length
    n = max_iterations

    S = initial_solution
    best_solution = S
    best_solution_distance = Calculate_total_distance(best_solution, distance_matrix)
    L = []
    L.append(S)
    for iteration in range(n):
        if iteration % 100 == 0:
            print(f"Iteration {iteration}")
        if len(L) > l:
            L.pop(0)
        R = Tweak_solution(copy.deepcopy(S), clusters, demands, vehicle_capacity, distance_matrix, weights)
        for _ in range(n - 1):
            W = Tweak_solution(copy.deepcopy(S), clusters, demands, vehicle_capacity, distance_matrix, weights)
            if W not in L and (Calculate_total_distance(W, distance_matrix) < Calculate_total_distance(R, distance_matrix) or R in L):
                R = W
        if R not in L:
            S = R
            L.append(R)
        if Calculate_total_distance(S, distance_matrix) < best_solution_distance:
            best_solution = S
            best_solution_distance = Calculate_total_distance(best_solution, distance_matrix)
    return best_solution, best_solution_distance

'Tabu Search implementation (FIFO for list) and reset tabu list at startover'
def Tabu_search_random_startover(initial_solution, clusters, demands, vehicle_capacity, distance_matrix, max_iterations, tabu_length, weights):
    l = tabu_length
    n = max_iterations
    K = len(initial_solution)

    S = initial_solution
    best_solution = S
    best_solution_distance = Calculate_total_distance(best_solution, distance_matrix)
    L = []
    L.append(S)
    for iteration in range(n):
        if iteration % 100 == 0 and iteration != 0:
            print(f"Iteration {iteration}: Random Startover")
            S = Random_initial_solution(nodes, clusters, demands, vehicle_capacity, K, distance_matrix)
            L = [S]
        if len(L) > l:
            L.pop(0)
        R = Tweak_solution(copy.deepcopy(S), clusters, demands, vehicle_capacity, distance_matrix, weights)
        for _ in range(n - 1):
            W = Tweak_solution(copy.deepcopy(S), clusters, demands, vehicle_capacity, distance_matrix, weights)
            if W not in L and (Calculate_total_distance(W, distance_matrix) < Calculate_total_distance(R, distance_matrix) or R in L):
                R = W
        if R not in L:
            S = R
            L.append(R)
        if Calculate_total_distance(S, distance_matrix) < best_solution_distance:
            best_solution = S
            best_solution_distance = Calculate_total_distance(best_solution, distance_matrix)
    return best_solution, best_solution_distance

############################################                MAIN                ############################################
filename = "C:/Users/mcpud/OneDrive/Documenten/Maastricht University/Master/C.gvrp.txt"
nodes, clusters, demands, vehicle_capacity, num_vehicles, distance_matrix = Read_gvrp_file(filename)
loop = 0
while loop < 4:
    """
    Since we implemented a local-search, tabu search to be more precise, method. We start by constructing an initialization method.
    - "Initial_solution()" is used to construct a solution that picks a cluster via nearest neighbor and uses the order of customers 
    given in the instance for the visiting order.
    - "Random_initial_solution()" is used to construct a random solution. This means choosing a random cluster and then assigning
    it to a random vehicle.
    """
    #initial_solution = Initial_solution(nodes, clusters, demands, vehicle_capacity, num_vehicles, distance_matrix)
    #initial_solution_distance = Calculate_total_distance(initial_solution, distance_matrix)
    #print("The initial solution is:")
    #print(initial_solution)
    #Print_solution(initial_solution, clusters)
    #print(f", with total distance: {initial_solution_distance}")
    random_initial_solution = Random_initial_solution(nodes, clusters, demands, vehicle_capacity, num_vehicles, distance_matrix)
    random_initial_solution_distance = Calculate_total_distance(random_initial_solution, distance_matrix)
    print("The initial solution is:")
    print(random_initial_solution)
    Print_solution(random_initial_solution, clusters)
    print(f", with total distance: {random_initial_solution_distance}")

    """
    Using the initial solution constructed via the method above, we start iteratively improving it. This is done via a tabu search.
    The parameters are set for the iterations, tabu list length and the weights for the Tweak operators. It uses three types of
    Tweak operators: Intra-route, Inter-route and Cluster reordering. 
    """

    iterations = 10000
    tabu_list_length = 10
    w_1, w_2, w_3 = 0.6, 0.2, 0.2
    weights = [w_1, w_2, w_3]
    best_solution, best_solution_distance = Tabu_search_random_startover(random_initial_solution, clusters, demands, vehicle_capacity, distance_matrix, iterations, tabu_list_length, weights)
    print(f"The best found solution with {iterations} iterations is:")
    Print_solution(best_solution, clusters)
    print(f", with a total distance of: {best_solution_distance} and weights {weights}")
    loop += 1