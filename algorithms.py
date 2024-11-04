from network_obj import * 
from collections import deque 
import heapq

'''
UNIFORM COST SEARCH / DJIKSTRAS / SPF FUNCTIONS
'''
# This is the uniform cost search function also known as shortest path first algorithm
def uniform_cost_search(goal, start):
    visited = set()
    queue = [(0, start, [])] # Cummulative cost, node 

    while queue:
        queue = sorted(queue)
        cost, node, path = heapq.heappop(queue) 
        # Skip if node has been visited 
        if node in visited:
            continue 
        visited.add(node) 
        current_path = path + [node]

        if node == goal:
            current_path.reverse()
            return current_path, cost 
        # Expand and add neighbors to priority queue
        for neighbor, path_cost in node.neighbors.items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + path_cost, neighbor, current_path)) 

    return None

'''
A STAR SEARCH FUNCTIONS
'''
# The A-Star Heuristic is to avoid high traffic intensity areas and high nodal delay areas
# The idea is that by avoiding congestion we will have lower transmission times
def a_star_heuristic(node, goal):
    for node in node.neighbors:
        if node == goal:
            return 0
    traffic_intensity = 10 * node.traffic 
    delay_penalty = node.delay
    return 1 + traffic_intensity + delay_penalty

def a_star_search(goal, start):
    frontier = [] 
    # Priority queue, reached dictionary 
    iter = 0 #iteration since priority queues don't allow duplicate nodes
    heapq.heappush(frontier, (0, iter, start)) 
    path_costs = {start: 0} # path cost seen dictionary
    path = {start: None} # path to a given node dictionary

    while frontier:
        # pop from queue. curr_score: f(n) = g(n) + h(n)
        _, _, curr_node = heapq.heappop(frontier) 
        if curr_node == goal:
            path = reconstruct_path(path, goal) 
            return path, path_costs[goal] 
        # for each neighbor of current node
        for neighbor, cost in curr_node.neighbors.items():
            temp_cost = path_costs[curr_node] + cost 
            
            # only update path if the lowest cost is seen 
            if neighbor not in path_costs or temp_cost < path_costs[neighbor]:
                path[neighbor] = curr_node 
                path_costs[neighbor] = temp_cost 
                f_score = temp_cost + a_star_heuristic(neighbor, goal) 
                iter += 1
                heapq.heappush(frontier, (f_score, iter, neighbor))
    return None # no path found

# reconstructs path found in a_star search
def reconstruct_path(paths, curr_node):
    path = [] 
    while curr_node:
        path.append(curr_node) 
        curr_node = paths[curr_node] 
    path.reverse() 
    return path 