from network_obj import *
from algorithms import *

# Set up for creating a network
def network1a():
    edges = {('A', 'B'): 9, ('A', 'C'): 13, ('B', 'C'): 10, ('B', 'D'): 8, ('C', 'E'): 5, ('D', 'E'): 3}
    nodes = {name: Node(name) for name in ['A', 'B', 'C', 'D', 'E']}
    config_graph(nodes, edges)
    nodes['A'].set_delay(5, 2) 
    nodes['B'].set_delay(3, 1)
    nodes['C'].set_delay(6, 1) 
    nodes['D'].set_delay(4, 1)
    nodes['E'].set_delay(5, 2) 
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
        print(str(d) + ", ", end=' ')
    print()
    for p in packets:
        print("Packet Details: " + str(p) + " Packet GID: " + str(p.gid) + " Delay (one way): " + str(p.delay) + "ms")

# Function for running a packet simulation through a network
def run_simulation(network, packet_cluster):
    print("Have entered function")
    # Setting up network
    nodes, edges = network() 
    print("After calling function in ")
    start, goal = nodes['A'], nodes['E'] 
    # Setting up packets
    group_ids = []
    packets = packet_cluster.copy()
    for p in packet_cluster:
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
    path, distance = a_star_search(start, goal) 
    print("A STAR SEARCH")
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
    packets.append(packet1)
    packets.append(packet2)
    run_simulation(network1a, packets)
