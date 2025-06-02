#!/usr/bin/env python3
"""
Genetic Algorithm for Traveling Salesman Problem (TSP)
Translated from C++ implementation in orig.cpp
"""

import random
import time
from typing import List, Tuple

def mutate(number_of_city: int, child: List[int], distance_matrix: List[List[float]]) -> List[int]:
    # print("Calling original mutate function")
    def objective_function(tour: List[int]) -> float:
        """Calculate the total distance of a tour (fitness function)."""
        total_distance = 0.0
        for i in range(len(tour) - 1):
            total_distance += distance_matrix[tour[i]][tour[i + 1]]
        # Add distance from last city back to first city
        total_distance += distance_matrix[tour[-1]][tour[0]]
        return total_distance

    def two_opt_swap(solution: List[int], i: int, j: int) -> List[int]:
        """2-opt swap operation."""
        new_solution = solution[:]
        new_solution[i:j+1] = reversed(new_solution[i:j+1])
        return new_solution

    def two_opt_shuffle(solution: List[int]) -> List[int]:
        """Random 2-opt shuffle operation."""
        tmp1 = random.randint(0, number_of_city - 1)
        tmp2 = random.randint(0, number_of_city - 1)
        
        while tmp2 < tmp1:
            tmp2 = random.randint(0, number_of_city - 1)
        
        new_solution = solution[:]
        segment = new_solution[tmp1:tmp2]
        random.shuffle(segment)
        new_solution[tmp1:tmp2] = segment
        return new_solution
    
    def two_opt_hill_climb(solution: List[int]) -> List[int]:
        """2-opt hill climbing local search."""
        improved = True
        best_solution = solution[:]
        best_distance = objective_function(solution)
        
        while improved:
            improved = False
            for i in range(number_of_city - 1):
                for j in range(i + 1, number_of_city):
                    new_solution = two_opt_swap(best_solution, i, j)
                    new_distance = objective_function(new_solution)
                    
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_solution = new_solution
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_solution

    if number_of_city <= 100:
        child = two_opt_hill_climb(child)
    else:
        child = two_opt_shuffle(child)
    return child

class GeneticAlgorithmTSP:
    """Genetic Algorithm implementation for solving TSP problems."""
    
    def __init__(self):
        # Global variables equivalent to C++ version
        self.distance_matrix: List[List[float]] = []
        self.population: List[List[int]] = []
        self.population_size: int = 0
        self.number_of_city: int = 0
        # self.gap: int = 0
        self.solution: float = 0.0
        self.percent: float = 0.0
        self.mutate_probability: float = 0.0
        self.initial_probability: float = 0.0
        self.select_probability: float = 0.0
        self.result_list: List[float] = []
    
    def generate_random_solution(self, num_cities: int) -> List[int]:
        """Generate a random solution by shuffling city indices."""
        solution = list(range(num_cities))
        random.shuffle(solution)
        return solution

    def two_opt_shuffle(self, solution: List[int]) -> List[int]:
        """Random 2-opt shuffle operation."""
        tmp1 = random.randint(0, self.number_of_city - 1)
        tmp2 = random.randint(0, self.number_of_city - 1)
        
        while tmp2 < tmp1:
            tmp2 = random.randint(0, self.number_of_city - 1)
        
        new_solution = solution[:]
        segment = new_solution[tmp1:tmp2]
        random.shuffle(segment)
        new_solution[tmp1:tmp2] = segment
        return new_solution
    
    def nearest_neighbor(self, start: int) -> List[int]:
        """Initialize solution using Nearest Neighbor heuristic."""
        n = len(self.distance_matrix)
        visited = [False] * n
        tour = []
        current = start
        
        tour.append(current)
        visited[current] = True
        
        for i in range(n - 1):
            min_dist = float('inf')
            nearest_neighbor = -1
            
            for j in range(n):
                if (not visited[j] and j != current and 
                    self.distance_matrix[current][j] < min_dist):
                    min_dist = self.distance_matrix[current][j]
                    nearest_neighbor = j
            
            if nearest_neighbor != -1:
                tour.append(nearest_neighbor)
                visited[nearest_neighbor] = True
                current = nearest_neighbor
        
        return tour
    
    def objective_function(self, tour: List[int]) -> float:
        """Calculate the total distance of a tour (fitness function)."""
        total_distance = 0.0
        for i in range(len(tour) - 1):
            total_distance += self.distance_matrix[tour[i]][tour[i + 1]]
        # Add distance from last city back to first city
        total_distance += self.distance_matrix[tour[-1]][tour[0]]
        return total_distance
    
    def tournament_selection(self, population: List[List[int]], 
                           tournament_size: int) -> List[int]:
        """Select best individual from a random tournament."""
        best_tour = []
        best_distance = float('inf')
        
        for _ in range(tournament_size):
            random_index = random.randint(0, len(population) - 1)
            candidate_tour = population[random_index]
            candidate_distance = self.objective_function(candidate_tour)
            
            if candidate_distance < best_distance:
                best_tour = candidate_tour[:]
                best_distance = candidate_distance
        
        return best_tour
    
    def order_crossover(self, parent1: List[int], 
                       parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX) for TSP."""
        first = random.randint(1, self.number_of_city // 2 - 1)
        second = random.randint(self.number_of_city // 2, self.number_of_city - 1)
        
        offspring1 = [None] * self.number_of_city
        offspring2 = [None] * self.number_of_city
        
        # Copy the selected segment
        for i in range(first, second + 1):
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]
        
        # Fill remaining positions
        it1, it2 = 0, 0
        for i in range(self.number_of_city):
            # For offspring1
            if parent2[i] not in offspring1 and it1 < self.number_of_city:
                offspring1[it1] = parent2[i]
                it1 += 1
                if it1 == first:
                    it1 = second + 1
            
            # For offspring2
            if parent1[i] not in offspring2 and it2 < self.number_of_city:
                offspring2[it2] = parent1[i]
                it2 += 1
                if it2 == first:
                    it2 = second + 1
        
        return offspring1, offspring2
    
    def initialize_population(self):
        """Initialize the population with random and nearest neighbor solutions."""
        self.population = []
        for i in range(self.population_size):
            if random.random() < self.initial_probability:
                tmp = self.nearest_neighbor(i % self.number_of_city)
            else:
                tmp = self.generate_random_solution(self.number_of_city)
            
            # Optional: Apply local search (commented out as in original)
            # if self.number_of_city <= 100:
            #     tmp = self.two_opt_hill_climb(tmp)
            # else:
            #     tmp = self.two_opt_shuffle(tmp)
            
            self.population.append(tmp)
    
    def generate_new_population(self, pop: List[List[int]]) -> List[List[int]]:
        """Generate new population through crossover and mutation."""
        new_population = []
        
        for _ in range(self.population_size):
            # Tournament selection
            parent1 = self.tournament_selection(self.population, 
                                              self.population_size // 2)
            parent2 = self.tournament_selection(self.population, 
                                              self.population_size // 2)
            
            # Crossover
            offspring1, offspring2 = self.order_crossover(parent1, parent2)
            
            # Select better offspring
            if self.objective_function(parent1) > self.objective_function(parent2):
                child = offspring1
            else:
                child = offspring2
            
            # Apply local search
            child = self.two_opt_shuffle(child)
            
            # Mutation
            x = random.random()
            if x < self.mutate_probability:
                child = mutate(self.number_of_city, child, self.distance_matrix)
            
            new_population.append(child)
        
        # Elitism: Keep better individuals
        for i in range(self.population_size):
            if (self.objective_function(pop[i]) < 
                self.objective_function(new_population[i])):
                new_population[i] = pop[i]
        
        return new_population
    
    def best_fitness(self) -> float:
        """Find the best fitness in current population."""
        best_distance = float('inf')
        for individual in self.population:
            distance = self.objective_function(individual)
            if distance < best_distance:
                best_distance = distance
        return best_distance
    
    def run(self) -> float:
        """Main genetic algorithm execution."""

        begin_time = time.time()
        # print("Running Genetic Algorithm for TSP...")
        # Set parameters based on problem size
        self.select_probability = 0.7
        
        self.gap = 20000
        if self.number_of_city <= 20:
            self.mutate_probability = 0.01
            self.initial_probability = 0.8
        elif self.number_of_city <= 50:
            self.mutate_probability = 0.01
            self.initial_probability = 0.85
        elif self.number_of_city <= 100:
            self.mutate_probability = 0.005
            self.initial_probability = 0.9
        else:
            self.mutate_probability = 0.005
            self.initial_probability = 1.0
        
        self.population_size = min(self.number_of_city // 2, 100)
        
        # Initialize population
        self.initialize_population()
        self.solution = self.best_fitness()
        self.result_list = [self.solution]
        
        # Evolution loop

        i = 0
        while (time.time() - begin_time < 9):
            self.population = self.generate_new_population(self.population)
            current_best = self.best_fitness()
            self.solution = min(current_best, self.solution)
            # print(f"Generation {i}: {current_best}")
            self.result_list.append(self.solution)
            i += 1

            if (i > self.gap and 
                self.result_list[i - self.gap] == self.result_list[i]):
                break
        
        return self.solution
    
    def load_problem(self):
        """Load TSP problem from file."""
        self.number_of_city = int(input())
        self.distance_matrix = []
        
        for i in range(self.number_of_city):
            row = list(map(float, input().strip().split()))
            self.distance_matrix.append(row)
    
    def load_problem_from_matrix(self, matrix: List[List[float]]):
        """Load TSP problem from distance matrix."""
        self.number_of_city = len(matrix)
        self.distance_matrix = [row[:] for row in matrix]


def main():
    """Main function for standalone execution."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create GA instance
    ga = GeneticAlgorithmTSP()
    
    # Load problem
    ga.load_problem()
    
    # Run algorithm
    best_solution = ga.run()
    
    print(f"{best_solution}")



if __name__ == "__main__":
    main()