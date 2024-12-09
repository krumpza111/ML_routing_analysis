from network_obj import *
from algorithms import *
import copy

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
    nodes['A'].set_delay(2, 2)
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
    nodes['A'].set_delay(2, 2)
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
    nodes['K'].set_delay(2, 2)
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

    #running genetic algorithms 
    path, distance = genetic_algorithm(nodes, start, goal) 
    print("GENETIC ALGORITHM")
    for id in group_ids:
        delays.append(packet_group_transmission(path, packets, id))
    print_results(path, packets, distance, delays)

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

    # running uniform cost search
    best_path, best_distance = a_star_search(start, goal)
    print("DYNAMIC A STAR SEARCH") 
    for id in group_ids:
        delays.append(dynamic_packet_group_transmission(best_path, packets, id))
        temp_path, temp_distance = a_star_search(start, goal)
        if temp_path != best_path:
            print("Better path found!")
            print("Old Distance " + str(best_distance) + " Old path: ", end="")
            print_path(best_path) 
            print("New distance: " + str(temp_distance) + " New path: ", end="")
            print_path(temp_path)
            best_path = temp_path 
            best_distance = temp_distance 
    print_results(best_path, packets, best_distance, delays)

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
    print("===========================================")
    print("          Running first network")
    print("===========================================")
    run_simulation(network1a, packets)
    print("===========================================")
    print("          Running second network")
    print("===========================================")
    run_simulation(network1b, packets)
    print("===========================================")
    print("          Running third network")
    print("===========================================")
    run_simulation(network2a, packets)
    print("===========================================")
    print("          Running fourth network")
    print("===========================================")
    run_simulation(network3a, packets)

    '''print("===========================================")
    print("      Running first dynamic network")
    print("===========================================")
    run_dynamic_simulation(network2a, packets)'''

