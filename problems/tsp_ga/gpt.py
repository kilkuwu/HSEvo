import random
from typing import List

def mutate_v2(
    number_of_city: int,
    child: List[int],
    distance_matrix: List[List[float]],
    max_size_for_exhaustive_2opt: int = 3187.6491165791276,
    max_bi_2opt_passes: int = 6.436451713265634,
    initial_ls_attempts_multiplier: float = 8.302200787905713,
    ils_ls_attempts_multiplier: float = 15.406485116248819,
    min_cities_for_4opt_perturb: int = 23.323912690002324,
    min_4opt_segment_ratio: float = 0.09417179701279878,
    max_perturb_selection_attempts: int = 97.49848908700983,
    min_cities_for_simple_perturb: int = 9.410434223186476,
    ils_cycles: int = 15.165754003861753,
    min_tour_len_for_ops: int = 1.0991060028991373,
    min_segment_abs_len: int = 1.139320746342055,
) -> List[int]:
    """
    Applies an adaptive Iterated Local Search (ILS) strategy for TSP mutation,
    combining best-improvement 2-opt for smaller instances with efficient stochastic 2-opt
    and a robust Double Bridge perturbation within ILS for larger networks.
    """
    n = number_of_city
    
    # Handle edge case for very small tours
    if n < min_tour_len_for_ops:
        return child[:] 

    # --- Helper Functions (optimized with O(1) delta calculations) ---

    def _get_delta_two_opt(tour: List[int], i: int, j: int) -> float:
        """
        Calculates the change in tour distance for a 2-opt swap between i and j.
        Achieves O(1) by only considering edges affected by the swap.
        Assumes i < j.
        """
        n_tour = len(tour)
        A = tour[(i - 1 + n_tour) % n_tour]
        B = tour[i]
        C = tour[j]
        D = tour[(j + 1) % n_tour]

        removed_dist = distance_matrix[A][B] + distance_matrix[C][D]
        added_dist = distance_matrix[A][C] + distance_matrix[B][D]
        return added_dist - removed_dist

    def _perform_two_opt_swap(tour: List[int], i: int, j: int) -> List[int]:
        """
        Executes a 2-opt swap by reversing the segment of the tour between i and j.
        Assumes i < j.
        """
        new_tour = tour[:]
        new_tour[i:j+1] = reversed(new_tour[i:j+1])
        return new_tour
    
    def _calculate_tour_distance(tour: List[int]) -> float:
        """Calculates the total distance of a given tour."""
        total_distance = 0.0
        n_tour = len(tour)
        for i in range(n_tour):
            total_distance += distance_matrix[tour[i]][tour[(i + 1) % n_tour]]
        return total_distance

    def _run_best_improvement_2opt(tour: List[int], max_passes: int) -> List[int]:
        """
        Performs best-improvement 2-opt hill climbing for a specified number of passes.
        Suitable for smaller networks where exhaustive search is feasible.
        """
        local_tour = tour[:]
        current_n = len(local_tour)
        iterations_without_improvement = 0

        while iterations_without_improvement < max_passes:
            found_improvement_in_pass = False
            best_local_delta = 0.0
            best_i, best_j = -1, -1

            for i in range(current_n - 1):
                # j must be at least i + 2 for a non-trivial 2-opt segment reversal
                for j in range(i + 2, current_n): 
                    delta = _get_delta_two_opt(local_tour, i, j)
                    if delta < best_local_delta:
                        best_local_delta = delta
                        best_i, best_j = i, j
                        found_improvement_in_pass = True
            
            if found_improvement_in_pass:
                local_tour = _perform_two_opt_swap(local_tour, best_i, best_j)
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
        return local_tour

    def _run_stochastic_best_2opt(tour: List[int], num_attempts: int) -> List[int]:
        """
        Applies stochastic best-improvement 2-opt. In each pass, it finds the best
        swap among random attempts and applies it. Repeats until no improvement in a pass.
        Efficient for larger networks.
        """
        local_tour = tour[:]
        current_n = len(local_tour)
        
        improvement_made_in_cycle = True
        while improvement_made_in_cycle:
            improvement_made_in_cycle = False
            best_delta_in_attempts = 0.0
            best_i_in_attempts, best_j_in_attempts = -1, -1

            if current_n < min_tour_len_for_ops: 
                break

            for _ in range(num_attempts):
                if current_n < 2: continue
                
                try:
                    i, j = random.sample(range(current_n), 2)
                except ValueError: # Not enough elements to sample
                    continue
                
                if i > j: i, j = j, i # Ensure i < j

                # Skip adjacent swaps (j == i + 1) as they are less effective in this context
                # and often implicitly handled by `j` in range(i+2, current_n) in best-improvement.
                if j == i + 1: continue 

                delta = _get_delta_two_opt(local_tour, i, j)
                if delta < best_delta_in_attempts:
                    best_delta_in_attempts = delta
                    best_i_in_attempts, best_j_in_attempts = i, j
                
            if best_delta_in_attempts < 0:
                local_tour = _perform_two_opt_swap(local_tour, best_i_in_attempts, best_j_in_attempts)
                improvement_made_in_cycle = True
        return local_tour

    def _perform_double_bridge_perturbation(tour: List[int]) -> List[int]:
        """
        Performs a Double Bridge (4-opt) perturbation for strong diversification.
        Includes robust segment length checks and fallbacks for smaller tours.
        """
        n_tour = len(tour)
        
        # Fallback for small networks where Double Bridge is not applicable/meaningful
        if n_tour < min_cities_for_4opt_perturb:
            if n_tour < min_cities_for_simple_perturb or n_tour < min_tour_len_for_ops: 
                return tour[:]
            
            # Perform a single random 2-opt swap as fallback
            if n_tour <= 2: 
                return tour[:]
            i, j = random.sample(range(n_tour), 2)
            if i > j: i, j = j, i
            if j == i + 1: return tour[:]
            return _perform_two_opt_swap(tour, i, j)

        min_segment_len = max(min_segment_abs_len, int(n_tour * min_4opt_segment_ratio))
        
        attempts = 0
        # Try to find four distinct points defining sufficiently long segments
        while attempts < max_perturb_selection_attempts:
            points = sorted(random.sample(range(n_tour), 4))
            p1, p2, p3, p4 = points[0], points[1], points[2], points[3]

            # Ensure segments between chosen points are long enough, including circular segment
            if (p2 - p1 >= min_segment_len and
                p3 - p2 >= min_segment_len and
                p4 - p3 >= min_segment_len and
                (n_tour - p4 + p1) >= min_segment_len):
                break
            attempts += 1
        
        # Fallback if valid points not found after max attempts
        if attempts == max_perturb_selection_attempts:
            if n_tour < min_cities_for_simple_perturb or n_tour < min_tour_len_for_ops: return tour[:]
            if n_tour <= 2: return tour[:]
            i, j = random.sample(range(n_tour), 2)
            if i > j: i, j = j, i
            if j == i + 1: return tour[:]
            return _perform_two_opt_swap(tour, i, j)

        # Apply the double bridge reordering: [S0 S1 S2 S3 S4] -> [S0 S3 S2 S1 S4]
        s0 = tour[0:p1+1]
        s1 = tour[p1+1:p2+1]
        s2 = tour[p2+1:p3+1]
        s3 = tour[p3+1:p4+1]
        s4 = tour[p4+1:n_tour]
        
        perturbed_tour = s0 + s3 + s2 + s1 + s4
        return perturbed_tour

    # --- Main mutate logic: Adaptive Iterated Local Search (ILS) ---
    current_tour = child[:]

    if n <= max_size_for_exhaustive_2opt:
        # For small to medium networks, perform a thorough best-improvement 2-opt.
        final_tour = _run_best_improvement_2opt(current_tour, max_passes=max_bi_2opt_passes)
    else:
        # For larger networks, use Iterated Local Search (ILS).
        
        # Step 1: Initial Local Search (Intensification)
        initial_2opt_attempts = int(n * initial_ls_attempts_multiplier)
        current_local_optimum = _run_stochastic_best_2opt(current_tour, num_attempts=initial_2opt_attempts)
        
        best_tour_so_far = current_local_optimum[:]
        best_distance_so_far = _calculate_tour_distance(best_tour_so_far)

        # Main ILS loop for multiple cycles
        for _ in range(int(ils_cycles)):
            # Step 2: Perturbation (Diversification) using Double Bridge
            perturbed_tour = _perform_double_bridge_perturbation(best_tour_so_far)
            
            if perturbed_tour == best_tour_so_far:
                continue

            # Step 3: Re-optimization (Intensification) with stochastic 2-opt
            re_opt_2opt_attempts = int(n * ils_ls_attempts_multiplier)
            re_optimized_tour = _run_stochastic_best_2opt(perturbed_tour, num_attempts=re_opt_2opt_attempts)
            re_optimized_distance = _calculate_tour_distance(re_optimized_tour)

            # Step 4: Acceptance Criterion (Elitist)
            # Accept only strictly better solutions to maintain quality.
            if re_optimized_distance < best_distance_so_far:
                best_tour_so_far = re_optimized_tour[:]
                best_distance_so_far = re_optimized_distance
            
        final_tour = best_tour_so_far
            
    return final_tour
