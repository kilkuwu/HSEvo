import random
import math
import time

class ACO():

    def __init__(self, 
                 distances,
                 heuristic,
                 decay=0.9,
                 alpha=1,
                 beta=1
                 ):
        
        self.problem_size = len(distances)
        self.distances = distances
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        # Set n_ants and n_iterations based on problem size (similar to GA)
        self.gap = 20000
        if self.problem_size <= 20:
            self.n_ants = 20
        elif self.problem_size <= 50:
            self.n_ants = 30
        elif self.problem_size <= 100:
            self.n_ants = 30
        else:
            self.n_ants = 20
            
        # Initialize pheromone matrix
        self.pheromone = [[1.0 for _ in range(self.problem_size)] for _ in range(self.problem_size)]
        self.heuristic = heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

    def run(self):
        begin_time = time.time()
        iteration = 0
        self.result_list = []
        while (time.time() - begin_time < 9):
            paths = self.gen_paths()
            costs = self.gen_path_costs(paths)
            # Find best path in this iteration
            best_cost = min(costs)
            best_idx = costs.index(best_cost)
            
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[best_idx][:]
                self.lowest_cost = best_cost

            
            self.update_pheromone(paths, costs)
            self.result_list.append(self.lowest_cost)

            i = iteration
            if (i > self.gap and 
                self.result_list[i - self.gap] == self.result_list[i]):
                break
            iteration += 1

        return self.lowest_cost
       
    def update_pheromone(self, paths, costs):
        '''
        Update pheromone based on paths and their costs
        Args:
            paths: list of paths, each path is a list of cities
            costs: list of costs corresponding to each path
        '''
        # Decay pheromone
        for i in range(self.problem_size):
            for j in range(self.problem_size):
                self.pheromone[i][j] *= self.decay
        
        # Add pheromone based on ant paths
        for ant_idx in range(self.n_ants):
            path = paths[ant_idx]
            cost = costs[ant_idx]
            pheromone_deposit = 1.0 / cost
            
            # Add pheromone to edges in the path
            for i in range(len(path)):
                from_city = path[i]
                to_city = path[(i + 1) % len(path)]  # Wrap around for TSP
                self.pheromone[from_city][to_city] += pheromone_deposit
                self.pheromone[to_city][from_city] += pheromone_deposit

    def gen_path_costs(self, paths):
        '''
        Calculate total distance for each path
        Args:
            paths: list of paths, each path is a list of cities
        Returns:
            costs: list of total distances
        '''
        costs = []
        for path in paths:
            total_cost = 0.0
            for i in range(len(path)):
                from_city = path[i]
                to_city = path[(i + 1) % len(path)]  # Wrap around for TSP
                total_cost += self.distances[from_city][to_city]
            costs.append(total_cost)
        return costs

    def gen_paths(self):
        '''
        Generate paths for all ants
        Returns:
            paths: list of paths, each path is a list of cities
        '''
        paths = []
        for ant in range(self.n_ants):
            path = self.gen_single_path()
            paths.append(path)
        return paths
        
    def gen_single_path(self):
        '''
        Generate a single path for one ant
        Returns:
            path: list of cities representing the tour
        '''
        # Start from a random city
        start_city = random.randint(0, self.problem_size - 1)
        path = [start_city]
        visited = [False] * self.problem_size
        visited[start_city] = True
        
        current_city = start_city
        
        # Build path by selecting next cities probabilistically
        for _ in range(self.problem_size - 1):
            next_city = self.pick_next_city(current_city, visited)
            path.append(next_city)
            visited[next_city] = True
            current_city = next_city
            
        return path
        
    def pick_next_city(self, current_city, visited):
        '''
        Pick next city based on pheromone and heuristic information
        Args:
            current_city: current city index
            visited: list of booleans indicating visited cities
        Returns:
            next_city: index of selected next city
        '''
        # Calculate probabilities for unvisited cities
        probabilities = []
        unvisited_cities = []
        
        for city in range(self.problem_size):
            if not visited[city]:
                pheromone_val = self.pheromone[current_city][city] ** self.alpha
                heuristic_val = self.heuristic[current_city][city] ** self.beta
                probability = pheromone_val * heuristic_val
                probabilities.append(probability)
                unvisited_cities.append(city)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob == 0:
            # If all probabilities are 0, choose randomly
            return random.choice(unvisited_cities)
        
        probabilities = [p / total_prob for p in probabilities]
        
        # Select city based on probabilities (roulette wheel selection)
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return unvisited_cities[i]
        
        # Fallback (should not reach here)
        return unvisited_cities[-1]