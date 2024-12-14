from network_obj import *
from algorithms import *
import copy

static_network_results = [] 

#class Simulation:

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

def dynamic_sim_cleanup(path, packets, distance, delays):
    print_results(path, packets, distance, delays)
    path = [] 
    reset_packets(packets)
    distance = 0
    delays = [] 

def distance_scrambler(edges):
    max_distance = max(edges.values())
    min_distance = min(edges.values())
    for edge in edges:
        rand_int = random.randrange(min_distance, (max_distance + 1))
        edges[edge] = rand_int 
    return edges

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
    
    delays = []

    # running uniform cost search 
    path, distance = uniform_cost_search(start, goal) 
    print("UNIFORM COST SEARCH") 
    for id in group_ids:
        delays.append(packet_group_transmission(path, packets, id))
    print_results(path, packets, distance, delays)
    delays = []
    reset_packets(packets) 

    #running a star search 
    path, distance = a_star_search(start, goal, a_star_heuristic) 
    print("A STAR SEARCH")
    for id in group_ids:
        delays.append(packet_group_transmission(path, packets, id))

    print_results(path, packets, distance, delays)
    delays = []
    reset_packets(packets)

    # A-Star search varient
    path, distance = a_star_search(start, goal, a_star_d_heuristic) 
    print("A STAR SEARCH WITH DISTANCE HEURISTIC") 
    for id in group_ids:
        delays.append(packet_group_transmission(path, packets, id))
    
    print_results(path, packets, distance, delays) 
    delays = [] 
    reset_packets(packets)

    #running greedy best first search 
    path, distance = gbfs(start, goal, gbfs_distance_heuristic)
    print("GREEDY BEST FIRST SEARCH") 
    for id in group_ids:
        delays.append(packet_group_transmission(path, packets, id)) 

    print_results(path, packets, distance, delays)
    delays = [] 
    reset_packets(packets)

    # GBFS search varient
    path, distance = gbfs(start, goal, gbfs_combined_heuristic)
    print("GREEDY BEST FIRST SEARCH WITH COMBINED HEURISTIC") 
    for id in group_ids:
        delays.append(packet_group_transmission(path, packets, id)) 

    print_results(path, packets, distance, delays) 
    delays = [] 
    reset_packets(packets)

    # Running Monte Carlo Tree Search 
    path, distance = mcts(start, goal)
    print("MONTE CARLO TREE SEARCH")
    for id in group_ids:
        delays.append(packet_group_transmission(path, packets, id)) 

    print_results(path, packets, distance, delays)
    delays = [] 
    reset_packets(packets)

    # Running CSPF Algorithm 
    path, distance = cspf_backtracking(start, goal, nodes) 
    print("CSPF ALGORITHM")
    for id in group_ids:
        delays.append(packet_group_transmission(path, packets, id))
    
    print_results(path, packets, distance, delays)
    delays = [] 
    reset_packets(packets)

    #running genetic algorithms 
    path, distance = genetic_algorithm(nodes, start, goal) 
    print("GENETIC ALGORITHM")
    for id in group_ids:
        delays.append(packet_group_transmission(path, packets, id))
    print_results(path, packets, distance, delays)

'''
DYNAMIC NETWORK SIMULATIONS
'''
# Function for running a packet simulation through a dynamic network
def run_dynamic_simulation(network, packet_cluster):
    nodes, edges = network()
    start, goal = nodes['A'], nodes['G']
    group_ids = [] 
    packet_cluster_copy = packet_cluster.copy() 
    packets = packet_cluster.copy() 
    for p in packet_cluster_copy:
        packets.extend(p.create_children()) 
        group_ids.append(p.gid)
    
    delays = [] 
    best_path = [] 
    best_distance = 0 

    # running UCS 
    best_path, best_distance = uniform_cost_search(start, goal)
    print("DYNAMIC UNIFORM COST SEARCH")
    for id in group_ids:
        delays.append(dynamic_packet_group_transmission(best_path, packets, id))
        temp_path, temp_distance = uniform_cost_search(start, goal) 
        if temp_path != best_path:
            print("New path found!")
            print("Old Distance " + str(best_distance) + " Old path: ", end="")
            print_path(best_path) 
            print("New distance: " + str(temp_distance) + " New path: ", end="")
            print_path(temp_path)
            best_path = temp_path 
            best_distance = temp_distance 
    print_results(best_path, packets, best_distance, delays)
    best_path = [] 
    reset_packets(packets)
    reset_traffic(nodes)
    best_distance = 0
    delays = [] 

    # running a-star search 
    best_path, best_distance = a_star_search(start, goal, a_star_heuristic)
    print("DYNAMIC A STAR SEARCH") 
    for id in group_ids:
        delays.append(dynamic_packet_group_transmission(best_path, packets, id))
        temp_path, temp_distance = a_star_search(start, goal, a_star_heuristic)
        if temp_path != best_path:
            print("New path found!")
            print("Old Distance " + str(best_distance) + " Old path: ", end="")
            print_path(best_path) 
            print("New distance: " + str(temp_distance) + " New path: ", end="")
            print_path(temp_path)
            best_path = temp_path 
            best_distance = temp_distance 
    print_results(best_path, packets, best_distance, delays)
    best_path = [] 
    reset_packets(packets)
    reset_traffic(nodes)
    best_distance = 0
    delays = [] 

    # running greedy best first search
    best_path, best_distance = gbfs(start, goal, gbfs_combined_heuristic) 
    print("DYNAMIC GREEDY BEST FIRST SEARCH")
    for id in group_ids:
        delays.append(dynamic_packet_group_transmission(best_path, packets, id)) 
        temp_path, temp_distance = gbfs(start, goal, gbfs_combined_heuristic) 
        if temp_path != best_path:
            print("New path found!")
            print("Old Distance " + str(best_distance) + " Old path: ", end="")
            print_path(best_path) 
            print("New distance: " + str(temp_distance) + " New path: ", end="")
            print_path(temp_path)
            best_path = temp_path 
            best_distance = temp_distance 
    print_results(best_path, packets, best_distance, delays)
    best_path = [] 
    reset_packets(packets)
    reset_traffic(nodes)
    best_distance = 0
    delays = [] 

    # running monte carlo tree search
    best_path, best_distance = mcts(start, goal)
    print("DYNAMIC MONTE CARLO TREE SEARCH")
    for id in group_ids:
        delays.append(dynamic_packet_group_transmission(best_path, packets, id))
        temp_path, temp_distance = mcts(start, goal)
        if temp_path != best_path:
            print("New path found!")
            print("Old Distance " + str(best_distance) + " Old path: ", end="")
            print_path(best_path) 
            print("New distance: " + str(temp_distance) + " New path: ", end="")
            print_path(temp_path)
            best_path = temp_path 
            best_distance = temp_distance 
    print_results(best_path, packets, best_distance, delays)
    best_path = [] 
    reset_packets(packets)
    reset_traffic(nodes)
    best_distance = 0
    delays = [] 

    # running CSPF
    best_path, best_distance = cspf_backtracking(start, goal, nodes) 
    print("DYNAMIC CSPF ALGORITHM")
    for id in group_ids:
        delays.append(dynamic_packet_group_transmission(best_path, packets, id)) 
        temp_path, temp_distance = cspf_backtracking(start, goal, nodes)
        if temp_path != best_path:
            print("New path found!")
            print("Old Distance " + str(best_distance) + " Old path: ", end="")
            print_path(best_path) 
            print("New distance: " + str(temp_distance) + " New path: ", end="")
            print_path(temp_path)
            best_path = temp_path 
            best_distance = temp_distance 
    print_results(best_path, packets, best_distance, delays) 
    best_path = [] 
    reset_packets(packets)
    reset_traffic(nodes)
    best_distance = 0
    delays = [] 

    # running Genetic Algorithm
    best_path, best_distance = genetic_algorithm(nodes, start, goal) 
    print("DYNAMIC GENETIC ALGORITHM")
    for id in group_ids:
        delays.append(dynamic_packet_group_transmission(best_path, packets, id)) 
        temp_path, temp_distance = genetic_algorithm(nodes, start, goal) 
        if temp_path != best_path:
            print("New path found!")
            print("Old Distance " + str(best_distance) + " Old path: ", end="")
            print_path(best_path) 
            print("New distance: " + str(temp_distance) + " New path: ", end="")
            print_path(temp_path)
            best_path = temp_path 
            best_distance = temp_distance 
    print_results(best_path, packets, best_distance, delays) 

'''
AD-HOC NETWORK SIMULATIONS
'''
def scramble_network(nodes, edges):
    if random.random() <= 0.33: # Flexibility (triggers 1/3rd of the time)
        print("Removing node") 
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
def run_ad_hoc_simulation(network, packet_cluster):
    nodes, edges = network() 
    print(nodes)
    #start, goal = nodes['A'], nodes['G']

    group_ids = [] 
    packet_cluster_copy = packet_cluster.copy() 
    packets = packet_cluster.copy() 
    for p in packet_cluster_copy:
        packets.extend(p.create_children()) 
        group_ids.append(p.gid)
    
    delays = [] 
    path = [] 
    distance = 0
    temp_nodes = nodes.copy()
    temp_edges = edges.copy()
    # Running UCS 
    print("AD-HOC UNIFORM COST SEARCH") 
    for id in group_ids:
        num_nodes = len(temp_nodes)
        temp_nodes, temp_edges = scramble_network(temp_nodes, temp_edges)
        print("After scrambling") 
        print(temp_edges)
        temp_num_nodes = len(temp_nodes.keys()) 
        start, goal = temp_nodes['A'], temp_nodes['G']
        if num_nodes == temp_num_nodes: # no nodes removed 
            temp_path, temp_distance = uniform_cost_search(start, goal) 
            delays.append(packet_group_transmission(temp_path, packets, id)) 
            if path is None:
                path = temp_path
                distance = temp_distance 
            else: # There is a path from before
                if temp_path != path:
                    print("New path found!")
                    print("Old Distance " + str(distance) + " Old path: ", end="")
                    print_path(path) 
                    print("New distance: " + str(temp_distance) + " New path: ", end="")
                    print_path(temp_path)
                    path =  temp_path
                    distance = temp_distance 
        else: # a node was removed 
            print("Node removed")
            temp_path, temp_distance = uniform_cost_search(start, goal) 
            delays.append(packet_group_transmission(temp_path, packets, id)) 
    print_results(path, packets, distance, delays)
    path = [] 
    reset_packets(packets)
    distance = 0
    delays = [] 

    # running A-star search 
    temp_nodes = nodes.copy() 
    temp_edges = edges.copy() 
    print("AD-HOC A STAR SEARCH")
    for id in group_ids:
        num_nodes = len(temp_nodes)
        temp_nodes, temp_edges = scramble_network(temp_nodes, temp_edges) 
        print("After scrambling") 
        print(temp_edges)
        temp_num_nodes = len(temp_nodes.keys()) 
        start, goal = temp_nodes['A'], temp_nodes['G']
        if num_nodes == temp_num_nodes:
            temp_path, temp_distance = a_star_search(start, goal, a_star_heuristic)
            delays.append(packet_group_transmission(temp_path, packets, id)) 
            if path is None:
                path = temp_path 
                distance = temp_distance 
            else:
                if temp_path != path:
                    print("New path found!")
                    print("Old Distance " + str(distance) + " | Old path: ", end="")
                    print_path(path) 
                    print("New distance: " + str(temp_distance) + " | New path: ", end="")
                    print_path(temp_path)
                    path = temp_path 
                    distance = temp_distance
        else:
            print("Node removed") 
            temp_path, temp_distance = a_star_search(start, goal, a_star_heuristic) 
            delays.append(packet_group_transmission(temp_path, packets, id))
    print_results(path, packets, distance, delays)
    path = [] 
    reset_packets(packets)
    distance = 0
    delays = [] 

    # running Greedy best first search
    temp_nodes = nodes.copy() 
    temp_edges = edges.copy() 
    print("AD-HOC GBFS")
    for id in group_ids:
        num_nodes = len(temp_nodes)
        temp_nodes, temp_edges = scramble_network(temp_nodes, temp_edges) 
        print("After scrambling") 
        print(temp_edges)
        temp_num_nodes = len(temp_nodes.keys()) 
        start, goal = temp_nodes['A'], temp_nodes['G']
        if num_nodes == temp_num_nodes:
            temp_path, temp_distance = gbfs(start, goal, gbfs_combined_heuristic)
            delays.append(packet_group_transmission(temp_path, packets, id)) 
            if path is None:
                path = temp_path 
                distance = temp_distance 
            else:
                if temp_path != path:
                    print("New path found!")
                    print("Old Distance " + str(distance) + " | Old path: ", end="")
                    print_path(path) 
                    print("New distance: " + str(temp_distance) + " | New path: ", end="")
                    print_path(temp_path)
                    path = temp_path 
                    distance = temp_distance
        else:
            print("Node removed") 
            temp_path, temp_distance = gbfs(start, goal, gbfs_combined_heuristic)
            delays.append(packet_group_transmission(temp_path, packets, id))
    print_results(path, packets, distance, delays)
    path = [] 
    reset_packets(packets)
    distance = 0
    delays = [] 

    # running Monte Carlo Tree Search
    temp_nodes = nodes.copy() 
    temp_edges = edges.copy() 
    print("AD-HOC MONTE CARLO")
    for id in group_ids:
        num_nodes = len(temp_nodes)
        temp_nodes, temp_edges = scramble_network(temp_nodes, temp_edges) 
        print("After scrambling") 
        print(temp_edges)
        temp_num_nodes = len(temp_nodes.keys()) 
        start, goal = temp_nodes['A'], temp_nodes['G']
        if num_nodes == temp_num_nodes:
            temp_path, temp_distance = mcts(start, goal)
            delays.append(packet_group_transmission(temp_path, packets, id)) 
            if path is None:
                path = temp_path 
                distance = temp_distance 
            else:
                if temp_path != path:
                    print("New path found!")
                    print("Old Distance " + str(distance) + " | Old path: ", end="")
                    print_path(path) 
                    print("New distance: " + str(temp_distance) + " | New path: ", end="")
                    print_path(temp_path)
                    path = temp_path 
                    distance = temp_distance
        else:
            print("Node removed") 
            temp_path, temp_distance = mcts(start, goal)
            delays.append(packet_group_transmission(temp_path, packets, id))
    print_results(path, packets, distance, delays)
    path = [] 
    reset_packets(packets)
    distance = 0
    delays = [] 
    
    # running CSPF
    temp_nodes = nodes.copy() 
    temp_edges = edges.copy() 
    print("AD-HOC CSPF")
    for id in group_ids:
        num_nodes = len(temp_nodes)
        temp_nodes, temp_edges = scramble_network(temp_nodes, temp_edges) 
        print("After scrambling") 
        print(temp_edges)
        temp_num_nodes = len(temp_nodes.keys()) 
        start, goal = temp_nodes['A'], temp_nodes['G']
        if num_nodes == temp_num_nodes:
            temp_path, temp_distance = cspf_backtracking(start, goal, temp_nodes)
            delays.append(packet_group_transmission(temp_path, packets, id)) 
            if path is None:
                path = temp_path 
                distance = temp_distance 
            else:
                if temp_path != path:
                    print("New path found!")
                    print("Old Distance " + str(distance) + " | Old path: ", end="")
                    print_path(path) 
                    print("New distance: " + str(temp_distance) + " | New path: ", end="")
                    print_path(temp_path)
                    path = temp_path 
                    distance = temp_distance
        else:
            print("Node removed") 
            temp_path, temp_distance = cspf_backtracking(start, goal, temp_nodes)
            delays.append(packet_group_transmission(temp_path, packets, id))
    print_results(path, packets, distance, delays)
    path = [] 
    reset_packets(packets)
    distance = 0
    delays = [] 

    # running Genetic Algorithm
    temp_nodes = nodes.copy() 
    temp_edges = edges.copy() 
    print("AD-HOC GENETIC ALGORITHMS")
    for id in group_ids:
        num_nodes = len(temp_nodes)
        temp_nodes, temp_edges = scramble_network(temp_nodes, temp_edges) 
        print("After scrambling") 
        print(temp_edges)
        temp_num_nodes = len(temp_nodes.keys()) 
        start, goal = temp_nodes['A'], temp_nodes['G']
        if num_nodes == temp_num_nodes:
            temp_path, temp_distance = genetic_algorithm(nodes, start, goal)
            delays.append(packet_group_transmission(temp_path, packets, id)) 
            if path is None:
                path = temp_path 
                distance = temp_distance 
            else:
                if temp_path != path:
                    print("New path found!")
                    print("Old Distance " + str(distance) + " | Old path: ", end="")
                    print_path(path) 
                    print("New distance: " + str(temp_distance) + " | New path: ", end="")
                    print_path(temp_path)
                    path = temp_path 
                    distance = temp_distance
        else:
            print("Node removed") 
            temp_path, temp_distance = genetic_algorithm(nodes, start, goal)
            delays.append(packet_group_transmission(temp_path, packets, id))

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
    for i in range(len(text)):
        print("===========================================")
        print("          Running " + text[i] + " network")
        print("===========================================")
        run_simulation(networks[i], packets)

    # Dynamic simulations
    for i in range(len(text)):
        print("===========================================")
        print("    Running " + text[i] + " dynamic network")
        print("===========================================")
        run_simulation(networks[i], packets)

    '''
    # Ad-Hoc Simulations
    for i in range(len(text)):
        print("===========================================")
        print("    Running " + text[i] + " ad-hoc network")
        print("===========================================")
        run_ad_hoc_simulation(networks[i], packets)
    '''


