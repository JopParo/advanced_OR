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
    d = {}  # Dictionary mapping edge (i,j) to distance
    
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
                        coords[node_id] = (x, y)
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
            for j in range(num_nodes):
                if i + j < len(lines):
                    parts = lines[i + j].strip().split()
                    if len(parts) >= 2:
                        node_id = int(parts[0])
                        demand = int(parts[1])
                        q[node_id] = demand
            i += num_nodes - 1
        i += 1
    
    # Create edges and calculate distances
    for i in V:
        for j in V:
            if i != j:
                E.append((i, j))
                # Calculate Euclidean distance
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                d[(i, j)] = distance
    
    # In the CluVRP, depot is typically not in any customer cluster
    # but is considered to be in its own special cluster r0
    r0 = 0  # Use 0 for depot cluster (assuming clusters are numbered from 1)
    # Make sure r0 is not already in R
    while r0 in R:
        r0 += 1
    R.append(r0)
    Cr[r0] = [V0]  # Create special cluster for depot
    
    # Prepare final result
    result = {
        'V': V,          # Set of vertices including depot and customer nodes
        'V0': V0,        # Depot node
        'E': E,          # Set of edges
        'K': K,          # Set of vehicles
        'Q': Q,          # Vehicle capacity
        'R': R,          # Set of clusters
        'Cr': Cr,        # Mapping of cluster index to list of customers
        'q': q,          # Mapping of customer index to demand
        'd': d,          # Distance matrix (as a dictionary)
        'r0': r0,        # Cluster containing only the depot
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
