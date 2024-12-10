from collections import deque

MAX_THROUGHPUT = 1024
BANDWIDTH = 10

# Class to create a node
# Each node is meant to represent a router so its attributes are:
# name =id, neighbors = connected nodes, delay = queueing + processing delay of that router 
# prop_delay = propogation delay, traffic = traffic intensity/congestion (ranges from 0 to 1)
class Node:
    def __init__(self, name):
        self.name = name 
        self.neighbors = {} # format is Node: {'cost': cost}
        self.delay = 0 
        self.prop_delay = 0
        self.traffic = 0
    
    def __lt__(self, other):
        return self.delay < other.delay

    # Creates adjacency list showing adjacent nodes and related cost
    def add_neighbor(self, neighbor, cost):
        self.neighbors[neighbor] = cost

    # Sets the nodes delay
    def set_delay(self, node_delay, prop_delay):
        self.delay = node_delay 
        self.prop_delay = prop_delay 

    # Sets the traffic intensity
    def set_traffic(self, n):
        # Must be bounded 
        if n > 1:
            n = 1 
        elif n < 0:
            n = 0 
        self.traffic = n 
    
    # resets traffic intensity to zero 
    def reset_traffic(self):
        self.traffic = 0

    # empties neigbors dictionary 
    def reset_neighbors(self):
        self.neighbors.clear()

    # Adds delay to packets as they pass through router
    def process_packet(self, packet, next_node):
        # process 
        if not isinstance(packet, Packet):
            return None
        throughput = 1.0
        if packet.size < MAX_THROUGHPUT:
            throughput = packet.size / MAX_THROUGHPUT
        if packet.first == True:
            if next_node == None:
                packet.delay += self.prop_delay
            else:
                path_cost = self.neighbors[next_node] 
                #print("path cost is " + str(path_cost) + " and the next node is " + str(next_node) + " with propogation delay " + str(next_node.prop_delay))
                packet.delay += (next_node.prop_delay + path_cost)
        packet.delay += (self.delay * throughput)
        return 
    
    # Adds delay and increases traffic intensity as packets pass through the network
    def process_packet_dyn(self, packet, next_node):
        if not isinstance(packet, Packet):
            return None
        throughput = 1.0 
        if packet.size < MAX_THROUGHPUT:
            throughput = packet.size / MAX_THROUGHPUT
        if packet.first == True:
            if next_node == None:
                packet.delay += self.prop_delay
            else:
                path_cost = self.neighbors[next_node] 
                packet.delay += (next_node.prop_delay + path_cost)
        self.traffic += 1 # increases traffic by one for each packet 
        packet.delay += (self.delay * throughput) + self.traffic
        return
    
    # print string
    def __str__(self):
        return f"{self.name}"

# Adds neigbors to all edges 
def config_graph(nodes, edge_list):
    for (source, dest), cost in edge_list.items():
        if source in nodes and dest in nodes:
            nodes[source].add_neighbor(nodes[dest], cost)
            nodes[dest].add_neighbor(nodes[source], cost)

# Class representing a packet
class Packet:
    # gid = group id 
    # size in bits (int)
    # first should be boolean indicating first packet 
    def __init__(self, gid, size, first):
        self.gid = gid 
        self.size = size 
        self.first = first
        self.delay = 0 

    # Splits up the packet to send through the network
    def create_children(self):
        children = [] 
        num_children = self.size // MAX_THROUGHPUT
        n_remaining = self.size % MAX_THROUGHPUT 
        self.size = MAX_THROUGHPUT
        for _ in range(num_children - 1):
            children.append(Packet(self.gid, MAX_THROUGHPUT, False)) 
        if n_remaining > 0:
            children.append(Packet(self.gid, n_remaining, False))
        return children 
    
    # Sets the accumulated delay of the packet back to zero
    def reset_delay(self):
        self.delay = 0
        
    # print string for the packet
    def __str__(self):
        return f"({self.gid})({self.size}){self.first}"

# simulates package transmission through a given path
def simulate_package_transmission(path, packet):
    total_delay = 0 
    for node in path:
        node.process_packet(packet)
        total_delay += packet.delay + node.traffic 

# Simulates transmission and adds delays of all packets together of a certatin group id
def packet_group_transmission(path, packets, gid):
    total_delay = 0
    for p in packets:
        if p.gid == gid:
            for i in range(len(path) - 1):
                path[i].process_packet(p, path[i + 1]) 
            path[len(path) - 1].process_packet(p, None)
            total_delay += p.delay
    return total_delay

# Simulates a dynamic transmission and adds delays of all packets of a certain group id
def dynamic_packet_group_transmission(path, packets, gid):
    total_delay = 0 
    for p in packets:
        if p.gid == gid:
            for i in range(len(path) - 1):
                path[i].process_packet_dyn(p, path[i + 1]) 
            path[len(path) - 1].process_packet_dyn(p, None)
            total_delay += p.delay 
    return total_delay

# resets all packets in a packet cluster
def reset_packets(packets):
    for p in packets:
        p.reset_delay() 
    
# resets all node traffic in the network
def reset_traffic(nodes):
    for node in nodes.values():
        node.reset_traffic()

# function that prints out a list of nodes in order
def print_path(path):
    if path is None:
        print("Error: path is empty")
        return
    string_builder = ""
    for node in path:
        if node == None:
            string_builder += " None "
            continue 
        if node == path[-1]:
            string_builder += str(node)
        else:
            string_builder += str(node) + " -> "
    print(string_builder)
