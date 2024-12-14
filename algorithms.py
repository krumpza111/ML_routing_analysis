from network_obj import * 
from collections import deque 
import heapq
import random
import math

BANDWIDTH = 10

'''
UNIFORM COST SEARCH / DJIKSTRAS / SPF FUNCTIONS
'''
# This is the uniform cost search function also known as shortest path first algorithm
def uniform_cost_search(start, goal):
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
            current_path
            return current_path, cost 
        # Expand and add neighbors to priority queue
        for neighbor, path_cost in node.neighbors.items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + path_cost, neighbor, current_path)) 

    return None

'''
A STAR SEARCH FUNCTIONS
A STAR SEARCH (Minimal delay and traffic intensity favored)
'''
# The A-Star Heuristic is to avoid high traffic intensity areas and high nodal delay areas
# The idea is that by avoiding congestion we will have lower transmission times
def a_star_heuristic(node, goal):
    for node in node.neighbors:
        if node == goal:
            return 0
    delay_penalty = node.delay
    return delay_penalty * node.traffic 

# A-star heuristic is to minimize distance first and then attempt to minimize the delay and traffic congestion experienced
def a_star_d_heuristic(node, goal):
    for n in node.neighbors:
        if n == goal:
            return 0
    neighbor_distance = min(distance for _, distance in node.neighbors.items()) 
    traffic_intensity = 1 + node.traffic / BANDWIDTH
    delay_penalty = node.delay
    return round((neighbor_distance * 0.7) + (traffic_intensity * 0.2) + (delay_penalty * 0.1))

def a_star_search(start, goal, heuristic):
    frontier = [] 
    # Priority queue, reached dictionary 
    iter = 0 #iteration since priority queues don't allow duplicate nodes
    heapq.heappush(frontier, (0, iter, start)) 
    path_costs = {start: 0} # path cost seen dictionary
    path = {start: None} # path to a given node dictionary
    f_scores = {start: heuristic(start, goal)}

    while frontier:
        # pop from queue. curr_score: f(n) = g(n) + h(n)
        _, _, curr_node = heapq.heappop(frontier) 
        if curr_node == goal:
            path = reconstruct_path(path, goal) 
            path.reverse()
            return path, f_scores[goal]
        # for each neighbor of current node
        for neighbor, cost in curr_node.neighbors.items():
            temp_cost = path_costs[curr_node] + cost 
            
            # only update path if the lowest cost is seen 
            if neighbor not in path_costs or temp_cost < path_costs[neighbor]:
                path[neighbor] = curr_node 
                path_costs[neighbor] = temp_cost 
                f_score = temp_cost + heuristic(neighbor, goal) 
                f_scores[neighbor] = f_score
                iter += 1
                heapq.heappush(frontier, (f_score, iter, neighbor))
    return None # no path found

# reconstructs path found in a_star search
def reconstruct_path(paths, curr_node):
    path = [] 
    while curr_node is not None:
        path.append(curr_node) 
        curr_node = paths[curr_node] 
    return path 

'''
GREEDY BEST FIRST SEARCH ALGORITHMS
'''
# Heuristic which minimizes average distance to neighbors
def gbfs_distance_heuristic(node, goal):
    if goal in node.neighbors:
        return node.neighbors[goal]
    min_dist = min(node.neighbors.values()) 
    return min_dist

# Heuristic which factors in distance and delay to minimize overall packet transmission
def gbfs_combined_heuristic(node, goal):
    traffic_intensity = 1 + node.traffic / BANDWIDTH 
    delay_penalty = node.delay 
    estimated_dist = min(node.neighbors.values()) 
    return estimated_dist + (delay_penalty + traffic_intensity)

# Greed best first search algorithm
def gbfs(start, goal, heuristic):
    fronteir = [] # Priority queue -- format: (heuristic_value, node)
    start_value = heuristic(start, goal)
    heapq.heappush(fronteir, (start_value, start)) # Add start node to the frontier
    reached = {start: None} # Dictionary lookup table 
    visited = set() # Set of visited nodes 
    while fronteir:
        _, curr_node = heapq.heappop(fronteir) 
        if curr_node in visited:
            # Skip already visited nodes
            continue
        visited.add(curr_node)

        if curr_node == goal:
            path = reconstruct_path(reached, goal) 
            path.reverse()
            total_dist = 0
            for i in range(len(path) - 1):
                curr_node = path[i]
                next_node = path[i + 1] 
                total_dist += curr_node.neighbors[next_node]
            return path, total_dist
        
        for neighbor, cost in curr_node.neighbors.items():
            if neighbor not in visited:
                reached[neighbor] = curr_node 
                cost_to_goal = heuristic(neighbor, goal)
                heapq.heappush(fronteir, (cost_to_goal, neighbor))
    return None

'''
CSPF
'''
def calculate_max_cost(nodes):
    max_cost = float('inf')
    for node in nodes.keys():
        for neighbor in nodes[node].neighbors.keys():
            temp_path_cost = nodes[node].neighbors[neighbor]
            total_cost = temp_path_cost + nodes[node].delay
            if total_cost > max_cost:
                max_cost = total_cost
    return max_cost 

def cspf_backtracking(start, goal, nodes):
    best_path = None
    best_cost = float('inf') 
    max_cost = calculate_max_cost(nodes)

    def dfs(curr_node, path, total_cost):
        nonlocal best_path, best_cost

        if curr_node == goal:
            if total_cost < max_cost:
                if total_cost < best_cost:
                    best_path = path[:]
                    best_cost = total_cost 
            return 
        
        for neighbor, cost in curr_node.neighbors.items():
            if neighbor in path:
                continue 
            new_cost = total_cost + cost 

            if new_cost <= max_cost:
                path.append(neighbor)
                dfs(neighbor, path, (total_cost + cost))
                path.pop() 
    dfs(start, [start], 0)
    return best_path, best_cost

'''
MONTE CARLO TREE SEARCH
'''

def mcts(start, goal):
    tree = {} # dictionary representing a tree. Format (state -> {visits, wins, children})
    exploration_factor = 1.4 # exploaration constant for UCT 
    hop_penalty = 0.2

    # Caclulates UCT value (upper confidence bound for trees)
    def uct_value(wins, visits, parent_visits):
        if visits == 0:
            return float('inf')
        return (wins / visits) + exploration_factor * math.sqrt(math.log(parent_visits) / visits)
    
    # Select best child using UCT
    def select(node):
        children = tree[node]["children"]
        parent_visits = tree[node]["visits"] 
        return max(children, key=lambda child: uct_value(tree[child]["wins"], tree[child]["visits"], parent_visits))
    
    # Expand tree by adding new child nodes
    def expand(node):
        possible_states = list(node.neighbors.keys()) 
        new_children = []
        for state in possible_states:
            if state not in tree:
                tree[state] = {"visits": 0, "wins": 0, "children": []}
                tree[node]["children"].append(state) 
                new_children.append(state)
        return new_children[0] if new_children else None
    
    # Simulate a random path from current node to a terminal state 
    def simulate(node):
        curr_node = node 
        total_cost = 0
        hop_count = 0
        visited = set()
        while curr_node != goal:
            visited.add(curr_node)
            neighbors = [neighbor for neighbor in curr_node.neighbors.keys() if neighbor not in visited] 
            if not neighbors:
                return float('inf') # Penalize dead end paths 
            next_node  = min(neighbors, key=lambda n: curr_node.neighbors[n]) # choose shortest closest neighbor
            total_cost += curr_node.neighbors[next_node] + curr_node.delay
            hop_count += 1
            curr_node = next_node
        # add delay of the node 
        total_cost += curr_node.delay
        total_cost += hop_count * hop_penalty
        return total_cost # returns total cost (distance) of path
    
    # Backpropogate results of simulation up fom path
    def backpropogate(path, result, reached_goal=False):
        for node in path:
            tree[node]["visits"] += 1 
            if reached_goal:
                tree[node]["wins"] += 10
            else:
                tree[node]["wins"] += 1

    # Function for reconstructing a path in MCTS
    def reconstruct_path():
        node = start 
        best_path = [start]
        while node != goal and tree[node]["children"]:
            node = max(tree[node]["children"], key=lambda child: uct_value(tree[child]["wins"], tree[child]["visits"], tree[node]["visits"]))
            best_path.append(node) 
        return best_path

    tree[start] = {"visits": 0, "wins": 0, "children": []} 
    # Perform simulations
    for _ in range(50):
        node = start 
        sim_path = [start]

        # Selection
        while node != goal and tree[node]["children"]:
            node = select(node)
            sim_path.append(node)
        
        # Expansion 
        if node != goal:
            new_node = expand(node)
            if new_node:
                sim_path.append(new_node) 
            
        # Simulation 
        result = simulate(sim_path[-1]) 

        # Backpropogation 
        reached_goal = sim_path[-1] == goal
        backpropogate(sim_path, result, reached_goal)

    # Reconstruct path to find optimal path 
    best_path = reconstruct_path()
    total_distance = sum(best_path[i].neighbors[best_path[i + 1]] for i in range(len(best_path) - 1))
    return best_path, total_distance

'''
Genetic Algorithm
General Set Up for the GA:
1. Represent route as chromosome (list of nodes) and randomly create set of initial routes 
2. Define fitness function based on path-cost and delay 
3. Select based on fitness 
4. Combine parts of better fitness paths and produce offspring 
5. Introduce random elements to offspring 
6. Stop after fixed number of generations or optimal route is found 
'''
# Class representing a path to the end goal and its total cost, and delay 
class Chromosome:
    def __init__(self, route, cost, delay):
        self.route = route
        self.cost = cost 
        self.delay = delay 
        self.fitness = self.calculate_fitness()
    
    def calculate_fitness(self):
        return 1 / (self.cost + self.delay + len(self.route))

# Caclculation of the routes total link-path cost, and total delay
def calculate_cost_delay(route):
    total_cost = 0 
    total_delay = 0 
    for i in range(len(route) - 1):
        curr_node = route[i]
        next_node = route[i + 1] 
        if next_node not in curr_node.neighbors:
            # Invalid paths are assigned maximum value (worst fitness) 
            return float('inf'), float('inf')
        total_cost += curr_node.neighbors[next_node] 
        total_delay += curr_node.delay 
        total_delay += next_node.delay 
    total_delay += route[-1].delay 
    return total_cost, total_delay 

# Creates the first generation population
def create_initial_population(nodes, start, goal, pop):
    population = [] 
    node_list = [] 
    for node in nodes.values():
        if node != start and node != goal:
            node_list.append(node) 
    #print_path(node_list)

    for _ in range(pop):
        subset_size = random.randint(1, len(node_list)) 
        random_subset = random.sample(node_list, subset_size)
        route = [start] + random_subset + [goal] 
        #print_path(route)

        valid_route = True 
        for i in range(len(route) - 1):
            if route[i + 1] not in route[i].neighbors:
                valid_route = False 
                break 
        if valid_route:
            # calculates and creates "individual" paths and puts them in the population
            cost, delay = calculate_cost_delay(route) 
            population.append(Chromosome(route, cost, delay))
    return population

# Function to select parents based on fitness
def select_parents(population):
    if not population:
        raise ValueError("Population is empty. Cannot select parents")
    
    total_fitness = sum(chromosome.fitness for chromosome in population) 
    fit_weights = [] 
    for chromosome in population:
        if chromosome.fitness == 0:
            fit_weights.append(0) 
        else:
            fit_weights.append(chromosome.fitness / total_fitness) 
    # Randomly selects chromosomes from population, weighted by fitness (chromosomes with higher fitness are picked more)
    parent1 = random.choices(population, fit_weights)[0]
    parent2 = random.choices(population, fit_weights)[0]
    return parent1, parent2

# Function to generate children 
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1.route) - 2) # Dont want to crossover the start or goal 
    child1_route = parent1.route[:crossover_point] + [node for node in parent2.route if node not in parent1.route[:crossover_point]]
    child2_route = parent2.route[:crossover_point] + [node for node in parent1.route if node not in parent2.route[:crossover_point]]

    cost1, delay1 = calculate_cost_delay(child1_route)
    cost2, delay2 = calculate_cost_delay(child2_route)

    child1 = Chromosome(child1_route, cost1, delay1)
    child2 = Chromosome(child2_route, cost2, delay2)
    return child1, child2

# Randomly mix up the nodes in the path to simulate mutations
def mutate(nodes, chromosome, mutation_rate):
    if random.random() < mutation_rate:
        mutation_type = random.choice(["swap", "add", "remove"]) 

        if mutation_type == "swap":
            if len(chromosome.route) > 3:
                i, j = random.sample(range(1, len(chromosome.route) - 1), 2)
                chromosome.route[i], chromosome.route[j] = chromosome.route[j], chromosome.route[i] 
        elif mutation_type == "add":
            available_nodes = [node for node in nodes.values() if node not in chromosome.route]
            if available_nodes:
                new_node = random.choice(available_nodes) 
                insert_position = random.randint(1, len(chromosome.route) - 1)
                chromosome.route.insert(insert_position, new_node)
        elif mutation_type == "remove":
            if len(chromosome.route) > 3:
                remove_position = random.randint(1, len(chromosome.route) - 2)
                chromosome.route.pop(remove_position)
        # update new chromosome
        cost, delay = calculate_cost_delay(chromosome.route) 
        chromosome.cost = cost 
        chromosome.delay = delay 
        chromosome.fitness = chromosome.calculate_fitness()

def genetic_algorithm(start, goal, nodes, pop=200, generations=100, init_mutation_rate=0.01):
    retry_count = 0 
    max_retries = 5
    while retry_count < max_retries:
        population = create_initial_population(nodes, start, goal, pop)
        if population:
            mutation_rate = init_mutation_rate 
            best_fitness = 0 
            stagnant_generations = 0
            solution = None

            for _ in range(generations):
                # Selection: Choose parents based on fitness
                parents = [select_parents(population) for _ in range(pop // 2)]

                # Crossover: Generate offspring 
                offspring = [] 
                for parent1, parent2 in parents:
                    child1, child2 = crossover(parent1, parent2)
                    offspring.append(child1)
                    offspring.append(child2)
        
                # Mutation: Apply random mutation to offspring
                for child in offspring:
                    mutate(nodes, child, mutation_rate) 

                population = population + offspring 
                population = sorted(population, key=lambda x: x.fitness, reverse=True)[:pop]
                best_solution = population[0] # best solution found

                if best_solution.fitness > best_fitness:
                    best_fitness = best_solution.fitness
                    solution = best_solution 
                    stagnant_generations = 0 
                else:
                    stagnant_generations += 1 

                # Adjust mutation rate if no improvement seen 
                if stagnant_generations > 15:
                    mutation_rate *= 1.2 
                else:
                    mutation_rate = max(init_mutation_rate, mutation_rate * 0.98) #gradually reduce mutation rate

            # Convert best_solution 
            path = [] 
            for node in solution.route:
                path.append(node)
            # return the path and distance
            return path, solution.cost
        else:
            retry_count += 1 
            pop += 50 
    raise ValueError(f"Failed to initialize valid population")


