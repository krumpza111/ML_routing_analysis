from collections import deque

MAX_THROUGHPUT = 1024

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
        self.traffic = 0.0

    def __lt__(self, other):
        return self.delay < other.delay

    # Creates adjacency list showing adjacent nodes and related cost
    def add_neighbor(self, neighbor, cost):
        self.neighbors[neighbor] = cost

    def set_delay(self, node_delay, prop_delay):
        self.delay = node_delay 
        self.prop_delay = prop_delay 

    def set_traffic(self, n):
        # Must be bounded 
        if n > 1:
            n = 1 
        elif n < 0:
            n = 0 
        self.traffic = n 

    # Adds delay to packets as they pass through router
    def process_packet(self, packet, next_node):
        # process 
        if not isinstance(packet, Packet):
            return None
        bandwidth = 1.0
        if packet.size < MAX_THROUGHPUT:
            bandwidth = packet.size / MAX_THROUGHPUT
        if packet.first == True:
            if next_node == None:
                packet.delay += self.prop_delay
            else:
                path_cost = self.neighbors[next_node] 
                #print("path cost is " + str(path_cost) + " and the next node is " + str(next_node) + " with propogation delay " + str(next_node.prop_delay))
                packet.delay += (next_node.prop_delay + path_cost)
        packet.delay += (self.delay * bandwidth)
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

# Simulates transmission and adds all packets together of a certatin group id
def packet_group_transmission(path, packets, gid):
    total_delay = 0
    for p in packets:
        if p.gid == gid:
            for i in range(len(path) - 1):
                path[i].process_packet(p, path[i + 1]) 
            path[len(path) - 1].process_packet(p, None)
            total_delay += p.delay
    return total_delay

# resets all packets in a packet cluster
def reset_packets(packets):
    for p in packets:
        p.reset_delay() 
    
def print_path(path):
    string_builder = ""
    for node in path:
        string_builder += str(node) + " "
    print(string_builder)
