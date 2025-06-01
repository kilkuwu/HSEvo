import random
import math

def mutate_v2(
    number_of_city: int,
    child: list,
    distance_matrix: list,
    *,  # Enforce keyword-only arguments for new parameters
    initial_or_opt_k_max: int = 4.679826002074293,
    final_or_opt_k_max: int = 2.874477382741107,
    min_cities_for_double_bridge: int = 15.404128077484794,
    small_problem_city_threshold: int = 117.45773445243238,
    medium_problem_city_threshold: int = 283.73911915636006,
    ils_iterations_medium_problem: int = 6.241273393641918,
    ils_iterations_large_problem: int = 4.7908314499360545,
) -> list:
    """
    Mutates a TSP tour using an evolved local search strategy with a focus on
    adaptive Iterated Local Search (ILS) and Variable Neighborhood Search (VNS) concepts.
    This version incorporates optimized local search operators (O(N^2) per pass)
    and adaptive perturbation, aiming for higher quality solutions.
    """

    def _objective_function(tour: list) -> float:
        """Calculate the total distance of a tour (fitness function)."""
        total_distance = 0.0
        for i in range(len(tour) - 1):
            total_distance += distance_matrix[tour[i]][tour[i + 1]]
        # Add distance from last city back to first city
        total_distance += distance_matrix[tour[-1]][tour[0]]
        return total_distance

    def _two_opt_swap_segment(solution: list, i: int, j: int) -> list:
        """Performs a 2-opt swap operation by reversing a segment of the tour."""
        new_solution = solution[:]
        # Ensure i < j for slicing
        if i > j:
            i, j = j, i
        new_solution[i:j+1] = reversed(new_solution[i:j+1])
        return new_solution

    def _two_opt_best_improvement_optimized(solution: list) -> list:
        """
        Performs 2-opt local search using 'best improvement' strategy with O(1) delta calculation per move.
        Total complexity: O(N^2) per pass. Iterates until no further improvement is found.
        """
        n = len(solution)
        current_solution = solution[:]
        
        while True:
            best_gain = 0.0
            best_i, best_j = -1, -1

            for i in range(n - 1):
                for j in range(i + 1, n):
                    city_i = current_solution[i]
                    city_i_plus_1 = current_solution[(i + 1) % n]
                    city_j = current_solution[j]
                    city_j_plus_1 = current_solution[(j + 1) % n] 

                    if (i + 1) % n == j: 
                        continue 

                    old_dist = distance_matrix[city_i][city_i_plus_1] + \
                               distance_matrix[city_j][city_j_plus_1]
                    new_dist = distance_matrix[city_i][city_j] + \
                               distance_matrix[city_i_plus_1][city_j_plus_1]
                    
                    gain = old_dist - new_dist

                    if gain > best_gain:
                        best_gain = gain
                        best_i, best_j = i, j
            
            if best_i != -1: 
                current_solution = _two_opt_swap_segment(current_solution, best_i, best_j)
            else: 
                break
        return current_solution

    def _or_opt_best_improvement_optimized(solution: list, k_max: int) -> list:
        """
        Performs Or-opt local search (relocation of segments of length 1, 2, or 3)
        using 'best improvement' strategy and O(1) delta cost calculation.
        Total complexity: O(k_max * N^2) per pass. Iterates until no improvement.
        """
        n = len(solution)
        current_solution = solution[:]

        while True:
            best_gain = 0.0
            best_reloc_i = -1 
            best_reloc_j = -1 
            best_reloc_k = -1 

            for k in range(1, min(k_max + 1, n)): # Segment length k from 1 to k_max
                for i in range(n): 
                    p_i_minus_1 = current_solution[(i - 1 + n) % n]
                    p_i = current_solution[i]
                    p_i_plus_k_minus_1 = current_solution[(i + k - 1) % n]
                    p_i_plus_k = current_solution[(i + k) % n]

                    removal_delta = distance_matrix[p_i_minus_1][p_i_plus_k] - \
                                    distance_matrix[p_i_minus_1][p_i] - \
                                    distance_matrix[p_i_plus_k_minus_1][p_i_plus_k]
                    
                    for j in range(n):
                        if (j == i) or (j == (i + k) % n) or (k == n):
                            continue 
                        
                        p_j_minus_1 = current_solution[(j - 1 + n) % n]
                        p_j = current_solution[j]

                        insertion_delta = distance_matrix[p_j_minus_1][p_i] + \
                                          distance_matrix[p_i_plus_k_minus_1][p_j] - \
                                          distance_matrix[p_j_minus_1][p_j]
                        
                        total_change = removal_delta + insertion_delta
                        gain = -total_change
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_reloc_i = i
                            best_reloc_j = j
                            best_reloc_k = k
            
            if best_reloc_i != -1: 
                new_solution = []
                segment_to_move = []
                
                for idx_offset in range(best_reloc_k):
                    segment_to_move.append(current_solution[(best_reloc_i + idx_offset) % n])
                
                temp_list_without_segment = []
                segment_indices_set = set([(best_reloc_i + x) % n for x in range(best_reloc_k)])
                for idx in range(n):
                    if idx not in segment_indices_set:
                        temp_list_without_segment.append(current_solution[idx])
                
                actual_insert_idx_in_temp = 0
                for original_idx in range(n):
                    if original_idx == best_reloc_j:
                        break
                    if original_idx not in segment_indices_set:
                        actual_insert_idx_in_temp += 1

                new_solution = temp_list_without_segment[:actual_insert_idx_in_temp] + \
                               segment_to_move + \
                               temp_list_without_segment[actual_insert_idx_in_temp:]

                current_solution = new_solution
            else: 
                break
        return current_solution

    def _double_bridge_kick(solution: list) -> list:
        """
        Performs a double-bridge kick perturbation. This is a 4-opt move that
        typically moves a solution out of a local optimum without drastically
        changing its structure. It breaks 4 edges and reconnects them.
        """
        n = len(solution)
        if n < min_cities_for_double_bridge: 
            return solution[:]

        while True:
            cut_points = sorted(random.sample(range(n), 4))
            x1, x2, x3, x4 = cut_points
            # Ensures each of the 5 segments (before, between cut points, after) are at least length 1.
            # This makes the cuts meaningful and avoids empty segments being swapped.
            if x1 >= 1 and x2 - x1 >= 1 and x3 - x2 >= 1 and x4 - x3 >= 1 and n - x4 >= 1:
                break 

        segment1 = solution[0:x1]
        segment2 = solution[x1:x2]
        segment3 = solution[x2:x3]
        segment4 = solution[x3:x4]
        segment5 = solution[x4:n]

        new_solution = segment1 + segment4 + segment3 + segment2 + segment5
        
        return new_solution

    # The main mutate_v2 function logic
    current_child = child[:]
    
    # Phase 1: Initial Strong Local Search (Optimized Best Improvement 2-Opt)
    current_child = _two_opt_best_improvement_optimized(current_child)

    # Phase 2: Adaptive ILS/VNS Strategy based on problem scale
    if number_of_city <= small_problem_city_threshold:
        current_child = _or_opt_best_improvement_optimized(current_child, k_max=initial_or_opt_k_max)
        current_child = _two_opt_best_improvement_optimized(current_child)

    elif number_of_city <= medium_problem_city_threshold:
        num_ils_iterations = ils_iterations_medium_problem 

        for _ in range(num_ils_iterations):
            perturbed_child = _double_bridge_kick(current_child)
            new_best_child = _two_opt_best_improvement_optimized(perturbed_child)
            
            if _objective_function(new_best_child) < _objective_function(current_child):
                current_child = new_best_child
        
        current_child = _or_opt_best_improvement_optimized(current_child, k_max=final_or_opt_k_max)

    else: # number_of_city > medium_problem_city_threshold
        num_ils_iterations = ils_iterations_large_problem 

        for _ in range(num_ils_iterations):
            perturbed_child = _double_bridge_kick(current_child)
            new_best_child = _two_opt_best_improvement_optimized(perturbed_child)
            
            if _objective_function(new_best_child) < _objective_function(current_child):
                current_child = new_best_child
        
        current_child = _or_opt_best_improvement_optimized(current_child, k_max=final_or_opt_k_max)
        
    return current_child
