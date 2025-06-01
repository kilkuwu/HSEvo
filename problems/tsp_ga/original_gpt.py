import random
from typing import List

def mutate_v2(number_of_city: int, child: List[int], distance_matrix: List[List[float]]) -> List[int]:
    """
    Evolves a tour (solution) using a combination of local search mutations:
    - Insertion: Move a city to a different position.
    - Reversion (2-opt): Reverse a section of the tour.
    - Displacement: Move a block of cities to another position.
    - Guided 2-opt hill climbing with adaptive perturbation based on city size.
    """

    def objective_function(tour: List[int]) -> float:
        """Calculates the total distance of a tour (fitness)."""
        total_distance = 0.0
        for i in range(len(tour) - 1):
            total_distance += distance_matrix[tour[i]][tour[i + 1]]
        total_distance += distance_matrix[tour[-1]][tour[0]]  # Return to start
        return total_distance

    def insertion_mutation(solution: List[int]) -> List[int]:
        """Move a city to a random new position."""
        idx1 = random.randint(0, number_of_city - 1)
        idx2 = random.randint(0, number_of_city - 1)
        city_to_move = solution[idx1]
        new_solution = solution[:idx1] + solution[idx1+1:]
        new_solution.insert(idx2, city_to_move)
        return new_solution

    def reversion_mutation(solution: List[int]) -> List[int]:
        """Reverse a section of the tour (2-opt)."""
        i = random.randint(0, number_of_city - 2)
        j = random.randint(i + 1, number_of_city - 1)
        new_solution = solution[:i] + list(reversed(solution[i:j + 1])) + solution[j + 1:]
        return new_solution

    def displacement_mutation(solution: List[int]) -> List[int]:
        """Move a block of cities to a new position."""
        length = random.randint(1, number_of_city // 4)  # Block size up to 25% of the tour
        start = random.randint(0, number_of_city - length)
        end = start + length
        block = solution[start:end]
        remaining = solution[:start] + solution[end:]
        insert_pos = random.randint(0, len(remaining))
        new_solution = remaining[:insert_pos] + block + remaining[insert_pos:]
        return new_solution
    
    def guided_2opt_hill_climb(solution: List[int], perturbation_rate: float = 0.05) -> List[int]:
        """2-opt hill climbing with guided search and perturbation."""
        best_solution = solution[:]
        best_distance = objective_function(solution)
        improved = True

        while improved:
            improved = False
            for i in range(number_of_city - 1):
                for j in range(i + 1, number_of_city):
                    new_solution = best_solution[:i] + list(reversed(best_solution[i:j + 1])) + best_solution[j + 1:]
                    new_distance = objective_function(new_solution)

                    if new_distance < best_distance:
                        best_solution = new_solution
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break

            #Perturbation to escape local optima: randomly apply a mutation
            if not improved and random.random() < perturbation_rate:
                mutation_type = random.choice([insertion_mutation, reversion_mutation, displacement_mutation])
                best_solution = mutation_type(best_solution)
                best_distance = objective_function(best_solution)  #Recompute distance
                improved = True  # Force at least one more hill-climbing iteration
        return best_solution

    #Adaptive strategy based on city count
    if number_of_city <= 50:
        #Smaller tours: Focus on thorough exploration
        child = guided_2opt_hill_climb(child, perturbation_rate=0.2) #Higher perturbation rate
    elif number_of_city <= 200:
        #Medium-sized tours: Balance exploitation and exploration
        child = guided_2opt_hill_climb(child, perturbation_rate=0.08)
    else:
        #Larger tours: More emphasis on efficient local search.
        child = guided_2opt_hill_climb(child, perturbation_rate=0.02) #Lower perturbation rate

    return child
