from network_obj import *
from algorithms import *
import copy
import time
from tabulate import tabulate

# Global tables to store results
static_network_results = {} 
dynamic_network_results = {} 
ad_hoc_network_results = {} 

# Set up for creating small-sized networks
def network1a():
    edges = {('A', 'B'): 9, ('A', 'C'): 13, ('B', 'C'): 10, ('B', 'D'): 8, ('C', 'G'): 5, ('D', 'G'): 3}
    nodes = {name: Node(name) for name in ['A', 'B', 'C', 'D', 'G']}
    config_graph(nodes, edges)
    nodes['A'].set_delay(5, 2) 
    nodes['B'].set_delay(3, 1)
    nodes['C'].set_delay(6, 1) 
    nodes['D'].set_delay(4, 1)
    nodes['G'].set_delay(5, 2) 
    return nodes, edges

def network1b():
    edges = {('A', 'B'): 1, ('A', 'G'): 12, ('B', 'C'): 1, ('C', 'G'): 2, ('B', 'D'): 2, ('D', 'E'): 3, ('C', 'E'): 1, ('E', 'G'): 3}
    nodes = {name: Node(name) for name in ['A', 'B', 'C', 'D', 'E', 'G']}
    config_graph(nodes, edges)
    nodes['A'].set_delay(4, 3)
    nodes['B'].set_delay(3, 2) 
    nodes['C'].set_delay(2, 1)
    nodes['D'].set_delay(3, 2)
    nodes['E'].set_delay(3, 1)
    nodes['G'].set_delay(4, 3)
    return nodes, edges 

# Set up for creating medium-sized networks
def network2a():
    edges = {('A', 'B'): 1, ('A', 'C'): 3, ('B', 'D'): 2, ('C', 'F'): 2, ('D', 'F'): 1, ('C', 'E'): 6, ('F', 'G'): 5, ('E', 'G'): 2}
    nodes = {name: Node(name) for name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']}
    config_graph(nodes, edges)
    nodes['A'].set_delay(3, 2)
    nodes['B'].set_delay(1, 2)
    nodes['C'].set_delay(3, 4)
    nodes['D'].set_delay(3, 1)
    nodes['E'].set_delay(4, 5)
    nodes['F'].set_delay(2, 1)
    nodes['G'].set_delay(3, 2)
    return nodes, edges

# Set up for large-sized networks 
def network3a():
    edges = {('A', 'B'): 4, ('A', 'C'): 7, ('A', 'D'): 9, ('B', 'C'): 8, ('B', 'H'): 5, ('C', 'H'): 3, ('D', 'H'): 6, ('D', 'E'): 10, ('E', 'F'): 2, ('H', 'J'): 5, ('H', 'I'): 15, ('H', 'F'): 6, ('J', 'K'): 4, ('J', 'I'): 3, ('K', 'L'): 6, ('K', 'I'): 5, ('I', 'L'): 3, ('E', 'G'): 17, ('F', 'G'): 10, ('I', 'G'): 5, ('L', 'G'): 7}
    nodes = {name: Node(name) for name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']}
    config_graph(nodes, edges)
    nodes['A'].set_delay(2, 2)
    nodes['B'].set_delay(3, 3)
    nodes['C'].set_delay(3, 2)
    nodes['D'].set_delay(3, 2)
    nodes['E'].set_delay(2, 1)
    nodes['F'].set_delay(2, 1)
    nodes['G'].set_delay(2, 2)
    nodes['H'].set_delay(4, 3)
    nodes['I'].set_delay(2, 2)
    nodes['J'].set_delay(2, 1)
    nodes['K'].set_delay(1, 1)
    nodes['L'].set_delay(1, 1)
    return nodes, edges

# Prints results after running a algorithm
def print_results(path, packets, distance, delays):
    print("Path: ", end='')
    for p in path:
        print(p, end=' ')
    print()
    print("Distance: ", distance)
    print("Total Nodal Delay: ", sum(delays))
    for d in delays:
        print(str(d) + " ", end=' ')
    print()
    for p in packets:
        print("Packet Details: " + str(p) + " Packet GID: " + str(p.gid) + " Delay (one way): " + str(p.delay) + "ms")

 # A function for assigning random distances to edges between the ranges of the highest and lowest distances  
 # Simulates mobility in ad-hoc networks     
def distance_scrambler(edges):
    max_distance = max(edges.values())
    min_distance = min(edges.values())
    for edge in edges:
        rand_int = random.randrange(min_distance, (max_distance + 1))
        edges[edge] = rand_int 
    return edges

# A function for assigning random delays to the nodes
# Simulates changing delays in ad-hoc networks that could be caused by "line-of-sight", interference, or a decrease in available bandwidth
def delay_scrambler(nodes):
    delays = [node.delay for node in nodes.values()]
    max_delay = max(delays)
    min_delay = min(delays)
    for node in nodes:
        rand_int = random.randrange(min_delay, (max_delay + 1))
        prop_delay = nodes[node].prop_delay
        nodes[node].set_delay(rand_int, prop_delay) 
    return nodes

'''
STATIC NETWORK SIMULATIONS
'''
# Function for running a packet simulation through a network
def run_algorithm(algorithm_name, start, goal, algorithm_function, group_ids, packets, *args):
    delays = [] 
    start_time = time.time() # start timer

    path, distance = algorithm_function(start, goal, *args) # run algorithm
    for id in group_ids:
        delays.append(packet_group_transmission(path, packets, id)) 

    end_time = time.time() # end timer
    total_time = end_time - start_time 

    # Recording results in table
    static_network_results[algorithm_name] = {'path': path, 'distance': distance, 'total_delay': sum(delays), 'total_time': total_time}
    reset_packets(packets)

# Function for running all algorithms within a certain network
def run_simulation(network, packet_cluster):
    # Setting up network
    nodes, edges = network() 
    start, goal = nodes['A'], nodes['G']

    # Setting up packets
    group_ids = []
    packet_cluster_copy = copy.deepcopy(packet_cluster)
    packets = copy.deepcopy(packet_cluster)
    for p in packet_cluster_copy:
        packets.extend(p.create_children()) 
        group_ids.append(p.gid)

    # Runnning all algorithms
    run_algorithm("UNIFORM COST SEARCH", start, goal, uniform_cost_search, group_ids, packets)
    run_algorithm("A STAR SEARCH", start, goal, a_star_search, group_ids, packets, a_star_heuristic)
    run_algorithm("A STAR (DISTANCE) SEARCH", start, goal, a_star_search, group_ids, packets, a_star_d_heuristic)
    run_algorithm("GREEDY BEST FIRST SEARCH", start, goal, gbfs, group_ids, packets, gbfs_distance_heuristic) 
    run_algorithm("MONTE CARLO TREE SEARCH", start, goal, mcts, group_ids, packets)
    run_algorithm("CSPF", start, goal, cspf_backtracking, group_ids, packets, nodes)
    run_algorithm("GENETIC ALGORITHM", start, goal, genetic_algorithm, group_ids, packets, nodes)

'''
DYNAMIC NETWORK SIMULATIONS
'''
# prints updated path change
def print_path_change(temp_path, best_path, temp_distance, best_distance):
    print("New path found!")
    print("Old Distance " + str(best_distance) + " Old path: ", end="")
    print_path(best_path) 
    print("New distance: " + str(temp_distance) + " New path: ", end="")
    print_path(temp_path)
    
# Function for testing a specfic algorithms routing 
def run_dynamic_algorithm(algorithm_name, start, goal, algorithm_function, nodes, group_ids, packets, *args):
    # tracking libraries
    delays = [] 
    best_path = [] 
    old_paths = {} 
    best_distance = 0 

    start_time = time.time() # start timer
    best_path, best_distance = algorithm_function(start, goal, *args) # Get initial paths for first packet
    for id in group_ids:
        delays.append(dynamic_packet_group_transmission(best_path, packets, id)) 
        temp_path, temp_distance = algorithm_function(start, goal, *args) # Run algorithm again to see if a better path is available
        if temp_path != best_path:
            #print_path_change(temp_path, best_path, temp_distance, best_distance)
            old_paths[print_path(temp_path)] = temp_distance
            best_path = temp_path 
            best_distance = temp_distance 

    end_time = time.time() # end timer
    total_time = end_time - start_time 

    # Recording results in table
    if len(old_paths) > 0:
        dynamic_network_results[algorithm_name] = {'path': best_path, 'distance': best_distance, 'total_delay': sum(delays), 'total_time': total_time, 'old_paths': old_paths}
    else:
        dynamic_network_results[algorithm_name] = {'path': best_path, 'distance': best_distance, 'total_delay': sum(delays), 'total_time': total_time, 'old_paths': "No change"}
    reset_packets(packets)
    reset_traffic(nodes)
    
# Function for running a packet simulation through a dynamic network
def run_dynamic_simulation(network, packet_cluster):
    # Setting up network
    nodes, edges = network()
    start, goal = nodes['A'], nodes['G']

    # Setting up packets
    group_ids = [] 
    packet_cluster_copy = packet_cluster.copy() 
    packets = packet_cluster.copy() 
    for p in packet_cluster_copy:
        packets.extend(p.create_children()) 
        group_ids.append(p.gid)
    
    # Running all algorithms
    run_dynamic_algorithm("UNIFORM COST SEARCH", start, goal, uniform_cost_search, nodes, group_ids, packets)
    run_dynamic_algorithm("A STAR SEARCH", start, goal, a_star_search, nodes, group_ids, packets, a_star_heuristic)
    run_dynamic_algorithm("A STAR (DISTANCE) SEARCH", start, goal, a_star_search, nodes, group_ids, packets, a_star_d_heuristic)
    run_dynamic_algorithm("GREEDY BEST FIRST SEARCH", start, goal, gbfs, nodes, group_ids, packets, gbfs_distance_heuristic)
    run_dynamic_algorithm("MONTE CARLO TREE SEARCH", start, goal, mcts, nodes, group_ids, packets)
    run_dynamic_algorithm("CSPF", start, goal, cspf_backtracking, nodes, group_ids, packets, nodes) 
    run_dynamic_algorithm("GENETIC ALGORITHM", start, goal, genetic_algorithm, nodes, group_ids, packets, nodes) 


'''
AD-HOC NETWORK SIMULATIONS
'''
# A function to simulate the randomness that can occur in ad-hoc networks
def scramble_network(nodes, edges):
    if random.random() <= 0.33: # Flexibility (triggers 1/3rd of the time)
        new_nodes, new_edges = reconfigure_graph(nodes, edges)
        for node in new_nodes:
            nodes[node].reset_neighbors() 
        config_graph(new_nodes, new_edges)
        return new_nodes, new_edges
    new_nodes = delay_scrambler(nodes)
    new_edges = distance_scrambler(edges) # Mobility
    for node in new_nodes:
        nodes[node].reset_neighbors() 
    config_graph(new_nodes, new_edges) 
    return new_nodes, new_edges

# Function for running a packet simulation through a ad-hoc network
def run_ad_hoc_algorithm(algorithm_name, start, goal, algorithm_function, nodes, edges, group_ids, packets, *args):
    # tracking libraries
    delays = [] 
    path = [] 
    distance = 0 
    old_paths = {}
    nodes_removed = []
    temp_nodes = nodes.copy() 
    temp_edges = edges.copy() 

    start_time = time.time() # start timer
    for id in group_ids:
        num_nodes = len(temp_nodes) 
        temp_nodes, temp_edges = scramble_network(temp_nodes, temp_edges)
        temp_num_nodes = len(temp_nodes.keys()) 
        start, goal = temp_nodes['A'], temp_nodes['G']
        if num_nodes == temp_num_nodes: # no nodes removed 
            temp_path, temp_distance = algorithm_function(start, goal, *args) 
            delays.append(packet_group_transmission(temp_path, packets, id)) 
            if path is None:
                path = temp_path
                distance = temp_distance 
            else: # There is a path from before
                if temp_path != path:
                    #print("New path found!")
                    #print("Old Distance " + str(distance) + " Old path: ", end="")
                    #print_path(path) 
                    #print("New distance: " + str(temp_distance) + " New path: ", end="")
                    #print_path(temp_path)
                    old_paths[print_path(temp_path)] = temp_distance
                    path =  temp_path
                    distance = temp_distance 
        else: # a node was removed 
            for node in nodes:
                if node not in temp_nodes and node not in nodes_removed:
                    nodes_removed.append(node)
            temp_path, temp_distance = algorithm_function(start, goal, *args) 
            delays.append(packet_group_transmission(temp_path, packets, id)) 

    end_time = time.time() # end timer
    total_time = end_time - start_time 

    # Recording results in table
    if len(old_paths) > 0:
        ad_hoc_network_results[algorithm_name] = {'path': path, 'distance': distance, 'total_delay': sum(delays), 'total_time': total_time, 'old_paths': old_paths, 'removed_nodes': nodes_removed} 
    else:
        ad_hoc_network_results[algorithm_name] = {'path': path, 'distance': distance, 'total_delay': sum(delays), 'total_time': total_time, 'old_paths': "No change", 'removed_nodes': nodes_removed}
    reset_packets(packets)

# Function that runs all algorithms through ad-hoc network simulations
def run_ad_hoc_simulation(network, packet_cluster):
    nodes, edges = network()
    start, goal = nodes['A'], nodes['G']
    group_ids = [] 
    packet_cluster_copy = packet_cluster.copy() 
    packets = packet_cluster.copy() 
    for p in packet_cluster_copy:
        packets.extend(p.create_children()) 
        group_ids.append(p.gid)

    # Running all algorithms
    run_ad_hoc_algorithm("UNIFORM COST SEARCH", start, goal, uniform_cost_search, nodes, edges, group_ids, packets)
    run_ad_hoc_algorithm("A STAR SEARCH", start, goal, a_star_search, nodes, edges, group_ids, packets, a_star_heuristic)
    run_ad_hoc_algorithm("A STAR (DISTANCE) SEARCH", start, goal, a_star_search, nodes, edges, group_ids, packets, a_star_d_heuristic)
    run_ad_hoc_algorithm("GREEDY BEST FIRST SEARCH", start, goal, gbfs, nodes, edges, group_ids, packets, gbfs_distance_heuristic)
    run_ad_hoc_algorithm("MONTE CARLO TREE SEARCH", start, goal, mcts, nodes, edges, group_ids, packets)
    run_ad_hoc_algorithm("CSPF", start, goal, cspf_backtracking, nodes, edges, group_ids, packets, nodes)
    run_ad_hoc_algorithm("GENETIC ALGORITHM", start, goal, genetic_algorithm, nodes, edges, group_ids, packets, nodes)

# Function for printing the results of the simulation into a neat table
def print_results():
    if static_network_results:
        headers = ["Algorithm", "Path", "Distance", "Delay", "Total Time"] 
        rows = [] 
        for alg, result in static_network_results.items():
            rows.append([alg, print_path(result['path']), result['distance'], result['total_delay'], result['total_time']]) 
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    if dynamic_network_results:
        headers = ["Algorithm", "Path", "Distance", "Delay", "Total Time", "Former Paths"]
        rows = [] 
        for alg, result in dynamic_network_results.items(): 
            rows.append([alg, print_path(result['path']), result['distance'], result['total_delay'], result['total_time'], result['old_paths']])
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    if ad_hoc_network_results:
        headers = ["Algorithm", "Path", "Distance", "Delay", "Total Time", "Former Paths", "Removed Nodes"]
        rows = []
        for alg, result in ad_hoc_network_results.items():
            rows.append([alg, print_path(result['path']), result['distance'], result['total_delay'], result['total_time'], result['old_paths'], result['removed_nodes']]) 
        print(tabulate(rows, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    '''
    General outline for running:
    nodes and edges are set by a network function 
    static network: only run once to get a path since traffic doesn't change 
    other networks: run search once per each packet to get a new path as traffic and nodes change
    '''
    packets = [] 
    packet1 = Packet(1, 5100, True) 
    packet2 = Packet(2, 2048, True)
    packet3 = Packet(3, 3871, True)
    packets.append(packet1)
    packets.append(packet2)
    packets.append(packet3) 

    networks = [network1a, network1b, network2a, network3a]
    text = ['First', 'Second', 'Third', "Fourth"]

    # Static Simulations
    print("============================================================")
    print("                  Running Static Networks")
    print("============================================================")
    for i in range(len(text)):
        print(" Running " + text[i] + " static network... ")
        run_simulation(networks[i], packets)
        print_results()
    static_network_results.clear()

    # Dynamic simulations
    print("============================================================")
    print("                  Running Dynamic Networks")
    print("============================================================")
    for i in range(len(text)):
        print()
        print(" Running " + text[i] + " dynamic network... ")
        print("============================================")
        run_dynamic_simulation(networks[i], packets)
        print_results()
    dynamic_network_results.clear()

    # Ad-Hoc Simulations
    print("============================================================")
    print("                  Running Ad-Hoc Networks")
    print("============================================================")
    for i in range(len(text)):
        print(" Running " + text[i] + " ad-hoc network... ")
        run_ad_hoc_simulation(networks[i], packets)
        print_results()



