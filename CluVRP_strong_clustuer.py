import math
import re
import numpy as np
import sys
import os

def read_gvrp_file(filename):
    """
    Read a .gvrp file and extract all relevant information for the CluVRP problem
    
    Parameters:
    filename (str): Path to the .gvrp file
    
    Returns:
    dict: Dictionary containing all problem parameters
    """
    # Initialize variables
    import numpy as np
    import math
    
    V = []  # Set of vertices (nodes)
    V0 = None  # Depot node
    E = []  # Set of edges
    K = []  # Set of vehicles
    Q = None  # Vehicle capacity
    R = []  # Set of clusters
    Cr = {}  # Dictionary mapping cluster index to list of customers in that cluster
    q = {}  # Dictionary mapping cluster index to demand
    
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Process header section
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if 'DIMENSION' in line:
            num_nodes = int(line.split(':')[1].strip())
        elif 'VEHICLES' in line:
            num_vehicles = int(line.split(':')[1].strip())
            K = list(range(num_vehicles))  # Create vehicle set
        elif 'GVRP_SETS' in line:
            num_clusters = int(line.split(':')[1].strip())
        elif 'CAPACITY' in line:
            Q = int(line.split(':')[1].strip())
        elif 'NODE_COORD_SECTION' in line:
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
                        coords[node_id] = (x, y)
                        V.append(node_id)  # Add to vertex set
            i += num_nodes - 1
            # First node is depot
            V0 = V[0]
        elif 'GVRP_SET_SECTION' in line:
            # Process clusters
            i += 1
            for j in range(num_clusters):
                if i + j < len(lines):
                    parts = lines[i + j].strip().split()
                    cluster_id = int(parts[0])
                    R.append(cluster_id)
                    
                    # Get customers in this cluster (excluding -1 terminator)
                    customers = [int(node) for node in parts[1:] if node != '-1']
                    Cr[cluster_id] = customers
            i += num_clusters - 1
        elif 'DEMAND_SECTION' in line:
            # Process demands - they correspond to clusters, not nodes
            i += 1
            for j in range(num_clusters):
                if i + j < len(lines):
                    parts = lines[i + j].strip().split()
                    if len(parts) >= 2:
                        cluster_id = int(parts[0])
                        demand = int(parts[1])
                        q[cluster_id] = demand
            i += num_clusters - 1
        
        i += 1

    # Create edges and calculate distances using a matrix
    # Find the maximum node ID to determine matrix size
    max_node_id = max(V)
    
    # Create a distance matrix with size [max_node_id+1 x max_node_id+1]
    # This allows direct indexing with node IDs
    d_matrix = np.full((max_node_id+1, max_node_id+1), np.inf)
    
    # Fill the distance matrix and create edges
    for i in V:
        for j in V:
            if i != j:
                E.append((i, j))
                # Calculate Euclidean distance
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                d_matrix[i, j] = distance
    
    # Set diagonal elements to 0 (distance from node to itself)
    np.fill_diagonal(d_matrix, 0)
    
    # Prepare final result
    result = {
        'V': V,          # Set of vertices including depot and customer nodes
        'V0': V0,        # Depot node
        'E': E,          # Set of edges
        'K': K,          # Set of vehicles
        'Q': Q,          # Vehicle capacity
        'R': R,          # Set of clusters
        'Cr': Cr,        # Mapping of cluster index to list of customers
        'q': q,          # Mapping of cluster index to demand
        'd': d_matrix,   # Distance matrix (as a numpy array)
        'coords': coords  # Store the coordinates for visualization if needed
    }
    
    return result

def main():
    # Change filename to desired file
    filename = "A.gvrp"
    problem_data = read_gvrp_file(filename)

    print(problem_data['V'])

if __name__ == "__main__":
    main()
