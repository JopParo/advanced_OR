import math
import numpy as np
import pandas as pd
import random

def read_gvrp_file(filename):
    """
    Read a .gvrp file and extract all relevant information for the CluVRP problem

    Parameters:
    filename (str): Path to the .gvrp file

    Returns:
    dict: Dictionary containing all problem parameters with variable names matching the document
    """
    # Initialize variables
    V = []  # Set of vertices (nodes)
    V0 = None  # Depot node
    E = []  # Set of edges
    K = []  # Set of vehicles
    Q = None  # Vehicle capacity
    R = []  # Set of clusters
    Cr = {}  # Dictionary mapping cluster index to list of customers in that cluster
    q = {}  # Dictionary mapping customer index to demand

    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Process header section
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('DIMENSION'):
            num_nodes = int(line.split(':')[1].strip())
        elif line.startswith('VEHICLES'):
            num_vehicles = int(line.split(':')[1].strip())
            K = list(range(num_vehicles))  # Create vehicle set
        elif line.startswith('GVRP_SETS'):
            num_clusters = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            Q = int(line.split(':')[1].strip())
        elif line.startswith('NODE_COORD_SECTION'):
            # Process node coordinates
            i += 1
            coords = {}
            for j in range(num_nodes):
                if i + j < len(lines):
                    parts = lines[i + j].strip().split()
                    if len(parts) >= 3:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        coords[node_id] = [x, y]
                        V.append(node_id)  # Add to vertex set
            i += num_nodes - 1
            # Assume first node is depot
            V0 = V[0]
        elif line.startswith('GVRP_SET_SECTION'):
            # Process clusters
            i += 1
            for _ in range(num_clusters):
                if i < len(lines):
                    parts = lines[i].strip().split()
                    cluster_id = int(parts[0])
                    R.append(cluster_id)

                    # Get customers in this cluster (excluding -1 terminator)
                    customers = [int(node) for node in parts[1:] if node != '-1']
                    Cr[cluster_id] = customers
                    i += 1
        elif line.startswith('DEMAND_SECTION'):
            # Process demands
            i += 1
            print(i)
            for j in range(num_clusters):
                if i + j < len(lines):
                    parts = lines[i + j].strip().split()
                    print(parts)
                    if len(parts) >= 2:
                        cluster_id = int(parts[0])
                        demand = int(parts[1])

                        # Assign demand to all customers in this cluster
                        if cluster_id in Cr:
                            for customer in Cr[cluster_id]:
                                q[customer] = demand
                        else:
                            print(f"Warning: Cluster {cluster_id} not found in Cr")

                        q[cluster_id] = demand
            i += num_clusters - 1
        i += 1

    d = pd.DataFrame(index=V)  # DataFrame assigning distance between any pair of nodes
    # Create edges and calculate distances
    for i in V:
        for j in V:
            if i == j:
                d.loc[i, j] = 0.0
            elif i != j:
                E.append((i, j))
                # Calculate Euclidean distance
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                d.loc[i, j] = distance
                d.loc[j, i] = distance

    # In the CluVRP, depot is typically not in any customer cluster
    # but is considered to be in its own special cluster r0
    r0 = 0  # Use 0 for depot cluster (assuming clusters are numbered from 1)
    # Make sure r0 is not already in R
    while r0 in R:
        r0 += 1
    R.append(r0)
    Cr[r0] = [V0]  # Create special cluster for depot

    # Calculate the mean coordinates of every cluster
    cluster_coordinates = {}
    for i in Cr:
        cluster = Cr[i]
        coordinates = []
        for customer in cluster:
            coordinates.append(coords[customer])
        coordinates = np.array(coordinates)
        cluster_coordinates[i] = np.average(coordinates, axis=0).tolist()

    # Calculate the average distance between each cluster
    cluster_distances = pd.DataFrame(index=R)  # DataFrame assigning distance between any pair of clusters
    for i in R:
        for j in R:
            if i == j:
                cluster_distances.loc[i, j] = 0.0
            elif i != j:
                # Calculate Euclidean distance
                x1, y1 = cluster_coordinates[i]
                x2, y2 = cluster_coordinates[j]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                cluster_distances.loc[i, j] = distance
                cluster_distances.loc[j, i] = distance

    # Prepare final result
    result = {
        'V': V,  # Set of vertices including depot and customer nodes
        'V0': V0,  # Depot node
        'E': E,  # Set of edges
        'K': K,  # Set of vehicles
        'Q': Q,  # Vehicle capacity
        'R': R,  # Set of clusters
        'Cr': Cr,  # Mapping of cluster index to list of customers
        'q': q,  # Mapping of customer index to demand
        'd': d,  # Distance matrix (as a DataFrame)
        'r0': r0,  # Cluster containing only the depot
        'coords': coords,  # Store the coordinates for visualization if needed
        'cluster_coordinates': cluster_coordinates, # Average coordinates of each cluster
        'cluster_distances': cluster_distances # Distance matrix between average coordinates of each cluster (as a DataFrame)
    }

    return result

def main():
    # Change filename to desired file
    filename = "/Users/yordivankruchten/Downloads/instances-set1/A.gvrp"
    problem_data = read_gvrp_file(filename)
    # print(f"Vertices: {problem_data['V']}")
    # print(f"Depot node: {problem_data['V0']}")
    # print(f"Edges: {problem_data['E']}")
    # print(f"Vehicles: {problem_data['K']}")
    # print(f"Vehicle capacity: {problem_data['Q']}")
    # print(f"Clusters: {problem_data['R']}")
    # print(f"Customer cluster index: {problem_data['Cr']}")
    # print(f"Customer demand: {problem_data['q']}")
    # print(f"Distance matrix {problem_data['d']}")
    # print(f"Depot cluster: {problem_data['r0']}")
    # print(f"Coordinates: {problem_data['coords']}")
    # print(f"Cluster coordinates: {problem_data['cluster_coordinates']}")
    # print(f"Distance matrix between clusters: {problem_data['cluster_distances']}")
    return problem_data

def results(problem_data, cluster_routes_per_vehicle, intra_cluster_routes):
    """
    Based on the cluster routes per vehicle and the intra-cluster routes,
    the total routes per vehicle are calculated. Then, the total distances of
    these routes are calculated.
    """
    vehicle_routes = {}
    for vehicle in cluster_routes_per_vehicle:
        vehicle_route = []
        cluster_route = cluster_routes_per_vehicle[vehicle]
        for cluster in cluster_route:
            vehicle_route.extend(intra_cluster_routes[cluster])
        vehicle_routes[vehicle] = vehicle_route
        vehicle_routes[vehicle].append(1)  # End the vehicle route at the depot

    vehicle_distances = {}
    for vehicle in vehicle_routes:
        distance = 0.0
        vehicle_route = vehicle_routes[vehicle]
        for i in range(len(vehicle_route) - 1):
            customer = vehicle_route[i]
            next_customer = vehicle_route[i + 1]
            distance += problem_data['d'].loc[customer, next_customer]
        vehicle_distances[vehicle] = float(distance)

    return vehicle_routes, vehicle_distances

def create_initial_solution(problem_data):
    """
    Assign clusters to vehicles by assigning the closest cluster to each vehicle's
    current location. This is a greedy assignment of clusters to vehicles to create
    an initial solution. This does not yet take into account vehicle capacities,
    because the demand object is empty somehow. Once, this issue is fixed, it should
    not be too difficult to implement this here. Subtract the total demand of a cluster
    from the remaining capacity of a vehicle, and only add the closest cluster, if this
    doesn't exceed the vehicle capacities. If it does, add the next closest cluster that
    doesn't exceed vehicle capacities if any.
    """
    clusters_to_assign = problem_data['R']
    cluster_routes_per_vehicle = {}
    for i in problem_data['K']:
        cluster_routes_per_vehicle[f"Vehicle {problem_data['K'][i]}"] = [problem_data['r0']]
    while len(clusters_to_assign) > 0:
        for i in cluster_routes_per_vehicle:
            current_location = cluster_routes_per_vehicle[i][-1]
            if current_location in clusters_to_assign:
                clusters_to_assign.remove(current_location)
            distances = {}
            if len(clusters_to_assign) > 0:
                for cluster in clusters_to_assign:
                    distances[cluster] = problem_data['cluster_distances'].loc[current_location, cluster]
                closest_cluster = min(distances, key=distances.get)
                cluster_routes_per_vehicle[i].append(closest_cluster)
                clusters_to_assign.remove(closest_cluster)

    """
    Simple nearest-neighbor heuristic for finding a route within a cluster. At first, 
    a random customer in a cluster is chosen. The route within this cluster starts at 
    this random customer. Starting from this customer, the nearest neighbor is found 
    and added to the route. This continues until all customers within a cluster are 
    added to the route. This results in a dictionary of nearest-neighbor intra-cluster 
    routes for all clusters that can be used for the initial solution.
    """
    intra_cluster_routes = {}
    for cluster in problem_data['Cr']:
        route = []
        customers = problem_data['Cr'][cluster]
        first_customer = random.choice(customers)
        route.append(first_customer)
        customers.remove(first_customer)
        while len(customers) > 0:
            distances = {}
            for customer in customers:
                distances[customer] = problem_data['d'].loc[route[-1], customer]
            next_customer = min(distances, key=distances.get)
            route.append(next_customer)
            customers.remove(next_customer)
        intra_cluster_routes[cluster] = route

    vehicle_routes, vehicle_distances = results(problem_data, cluster_routes_per_vehicle, intra_cluster_routes)

    return cluster_routes_per_vehicle, intra_cluster_routes, vehicle_routes, vehicle_distances

problem_data = main()
initial_cluster_routes_per_vehicle, initial_intra_cluster_routes, initial_vehicle_routes, initial_vehicle_distances = create_initial_solution(problem_data)

print(f"Cluster routes per vehicle: {initial_cluster_routes_per_vehicle}")
print(f"Intra-cluster routes: {initial_intra_cluster_routes}")
print(f"Vehicle routes: {initial_vehicle_routes}")
print(f"Total distance per vehicle: {initial_vehicle_distances}")