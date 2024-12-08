from network_obj import * 
from collections import deque 
import heapq
import random


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
'''
# The A-Star Heuristic is to avoid high traffic intensity areas and high nodal delay areas
# The idea is that by avoiding congestion we will have lower transmission times
def a_star_heuristic(node, goal):
    for node in node.neighbors:
        if node == goal:
            return 0
    traffic_intensity = 1 + node.traffic / BANDWIDTH
    delay_penalty = node.delay
    return delay_penalty * node.traffic 

def a_star_search(start, goal):
    frontier = [] 
    # Priority queue, reached dictionary 
    iter = 0 #iteration since priority queues don't allow duplicate nodes
    heapq.heappush(frontier, (0, iter, start)) 
    path_costs = {start: 0} # path cost seen dictionary
    path = {start: None} # path to a given node dictionary
    f_scores = {start: a_star_heuristic(start, goal)}

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
                f_score = temp_cost + a_star_heuristic(neighbor, goal) 
                f_scores[neighbor] = f_score
                iter += 1
                heapq.heappush(frontier, (f_score, iter, neighbor))
    return None # no path found

# reconstructs path found in a_star search
def reconstruct_path(paths, curr_node):
    path = [] 
    while curr_node:
        path.append(curr_node) 
        curr_node = paths[curr_node] 
    return path 

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

def genetic_algorithm(nodes, start, goal, pop=150, generations=20, init_mutation_rate=0.01):
    population = create_initial_population(nodes, start, goal, pop)
    mutation_rate = init_mutation_rate 
    best_fitness = 0 
    stagnant_generations = 0
    solution = None

    for generation in range(generations):
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


