import random
from typing import List

def mutate_v2(number_of_city: int, child: List[int], distance_matrix: List[List[float]]) -> List[int]:
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

