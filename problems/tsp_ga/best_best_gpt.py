import random
import math
from typing import List

def mutate_v2(number_of_city: int, child: List[int], distance_matrix: List[List[float]]) -> List[int]:
    """
    Implements an advanced Iterated Local Search (ILS) mutation for TSP.
    It combines Variable Neighborhood Descent (VND) with a diverse pool of perturbation
    operators, adaptive operator selection (AOS), and robust stagnation handling,
    including a crucial adaptive cap on perturbation intensity for scalability.
    """
    if number_of_city <= 1:
        return child[:]

    # --- Helper functions ---

    def _calculate_tour_distance(tour: List[int], dm: List[List[float]]) -> float:
        """Calculates the total distance of a given TSP tour."""
        total_distance = 0.0
        n = len(tour)
        if n <= 1:
            return 0.0
        for i in range(n - 1):
            total_distance += dm[tour[i]][tour[i + 1]]
        total_distance += dm[tour[-1]][tour[0]]  # Return to start
        return total_distance

    def _two_opt_swap(solution: List[int], i: int, j: int) -> List[int]:
        """
        Performs a 2-opt swap operation by reversing a segment of the tour.
        Ensures i < j for valid slicing and reversal. The segment reversed is from index i to j (inclusive).
        """
        new_solution = solution[:]
        if i > j:
            i, j = j, i
        new_solution[i:j+1] = new_solution[i:j+1][::-1]
        return new_solution

    def _is_index_in_segment(idx_to_check: int, s_start: int, s_end: int, total_n: int) -> bool:
        """Helper for Or-Opt to check if an index is within a (possibly wrapping) segment."""
        if s_start <= s_end:
            return s_start <= idx_to_check <= s_end
        else: # Segment wraps around (e.g., [n-2, n-1, 0, 1])
            return idx_to_check >= s_start or idx_to_check <= s_end

    # --- Local Search Operators (Best Improvement) ---

    def _best_improvement_2_opt(initial_solution: List[int], dm: List[List[float]]) -> List[int]:
        """
        Performs 2-opt local search using a 'best improvement' strategy.
        Optimized using delta cost calculation for O(N^2) complexity.
        """
        current_solution = initial_solution[:]
        n = len(current_solution)
        
        if n < 3:
            return current_solution

        while True:
            best_improvement_found_in_pass = 0.0
            best_i, best_j = -1, -1

            for i in range(n):
                for j in range(i + 1, n):
                    if j == (i + 1) % n or (i == 0 and j == n-1): # Avoid adjacent or wrapping segments that are effectively 0-length swap
                        continue
                    
                    node_prev_i = current_solution[(i - 1 + n) % n]
                    node_i = current_solution[i]
                    node_j = current_solution[j]
                    node_next_j = current_solution[(j + 1) % n]

                    old_edges_cost = dm[node_prev_i][node_i] + \
                                     dm[node_j][node_next_j]
                    new_edges_cost = dm[node_prev_i][node_j] + \
                                     dm[node_i][node_next_j]
                    
                    current_improvement = old_edges_cost - new_edges_cost

                    if current_improvement > best_improvement_found_in_pass:
                        best_improvement_found_in_pass = current_improvement
                        best_i, best_j = i, j
            
            if best_improvement_found_in_pass > 1e-9:
                current_solution = _two_opt_swap(current_solution, best_i, best_j)
            else:
                break
        return current_solution

    def _best_improvement_or_opt(initial_solution: List[int], dm: List[List[float]]) -> List[int]:
        """
        Performs Or-opt local search using 'best improvement' with O(N^2) delta cost,
        handling circular tours for complete neighborhood exploration. Moves a segment of 1, 2, or 3 cities.
        """
        current_solution = initial_solution[:]
        n = len(current_solution)
        if n < 3:
            return current_solution

        max_segment_length_for_oropt = min(3, n - 1)

        while True:
            best_improvement_found_in_pass = 0.0
            best_segment_start_idx, best_segment_end_inclusive_idx, best_insert_pos_after = -1, -1, -1

            for segment_len in range(1, max_segment_length_for_oropt + 1):
                for i in range(n):  # i is the start index of the segment to move
                    j_inclusive = (i + segment_len - 1) % n  # j_inclusive is the end index of the segment

                    node_before_segment = current_solution[(i - 1 + n) % n]
                    node_first_in_segment = current_solution[i]
                    node_last_in_segment = current_solution[j_inclusive]
                    node_after_segment = current_solution[(j_inclusive + 1) % n]

                    for k_pos_before_insertion in range(n):  # k_pos_before_insertion is the index of the node *before* the insertion point
                        # Skip if insertion point is within the segment or adjacent to its original location
                        if _is_index_in_segment(k_pos_before_insertion, i, j_inclusive, n) or \
                           _is_index_in_segment((k_pos_before_insertion + 1) % n, i, j_inclusive, n) or \
                           k_pos_before_insertion == (i - 1 + n) % n: 
                            continue

                        node_at_insert_before = current_solution[k_pos_before_insertion]
                        node_at_insert_after = current_solution[(k_pos_before_insertion + 1) % n]

                        # Calculate delta cost for removing segment from original place
                        cost_removed_original_edges = dm[node_before_segment][node_first_in_segment] + \
                                                      dm[node_last_in_segment][node_after_segment]
                        cost_added_after_removal = dm[node_before_segment][node_after_segment]
                        
                        # Calculate delta cost for adding segment to new place
                        cost_removed_insertion_point_edge = dm[node_at_insert_before][node_at_insert_after]
                        cost_added_at_insertion_point = dm[node_at_insert_before][node_first_in_segment] + \
                                                        dm[node_last_in_segment][node_at_insert_after]
                        
                        current_improvement = (cost_removed_original_edges - cost_added_after_removal) + \
                                              (cost_removed_insertion_point_edge - cost_added_at_insertion_point)
                        
                        if current_improvement > best_improvement_found_in_pass:
                            best_improvement_found_in_pass = current_improvement
                            best_segment_start_idx = i
                            best_segment_end_inclusive_idx = j_inclusive
                            best_insert_pos_after = k_pos_before_insertion

            if best_improvement_found_in_pass > 1e-9:
                # Apply the best found Or-opt move
                segment_to_move = []
                if best_segment_start_idx <= best_segment_end_inclusive_idx:
                    segment_to_move = current_solution[best_segment_start_idx : best_segment_end_inclusive_idx + 1]
                else: # Segment wraps around
                    segment_to_move = current_solution[best_segment_start_idx:] + current_solution[:best_segment_end_inclusive_idx + 1]
                
                # Create the remaining tour without the segment, handling wrap-around for the "removed" part
                remaining_tour = []
                if best_segment_start_idx <= best_segment_end_inclusive_idx:
                    remaining_tour = current_solution[:best_segment_start_idx] + current_solution[best_segment_end_inclusive_idx + 1:]
                else: 
                    remaining_tour = current_solution[best_segment_end_inclusive_idx + 1 : best_segment_start_idx]
                
                # Determine the actual insertion point in the remaining_tour list
                insert_idx_in_remaining_tour = 0
                if len(remaining_tour) > 0:
                    try:
                        original_node_before_insertion = current_solution[best_insert_pos_after]
                        insert_idx_in_remaining_tour = remaining_tour.index(original_node_before_insertion) + 1
                    except ValueError:
                        insert_idx_in_remaining_tour = len(remaining_tour)
                
                insert_idx_in_remaining_tour = max(0, min(insert_idx_in_remaining_tour, len(remaining_tour)))

                current_solution = remaining_tour[:insert_idx_in_remaining_tour] + segment_to_move + remaining_tour[insert_idx_in_remaining_tour:]
            else:
                break
        return current_solution

    def _best_improvement_3_opt_extended(initial_solution: List[int], dm: List[List[float]]) -> List[int]:
        """
        Performs 3-opt local search by evaluating multiple 3-opt move types (A-C'-B'-D, A-B'-C'-D, A-C-B-D).
        Uses O(1) delta cost calculations for each variant, maintaining O(N^3) complexity.
        """
        current_solution = initial_solution[:]
        n = len(current_solution)
        if n < 4:
            return current_solution

        while True:
            best_improvement_found_in_pass = 0.0
            best_i, best_j, best_k = -1, -1, -1
            best_variant = -1 

            for i_node in range(n):
                for j_node in range(i_node + 1, n):
                    for k_node in range(j_node + 1, n):
                        # Ensure three distinct non-adjacent edges are chosen (i, i+1), (j, j+1), (k, k+1)
                        # We use modular arithmetic for nodes to handle wrap-around, simplifying index logic.
                        # This also ensures the segments are non-empty.
                        
                        # Indices of the first node of each broken edge
                        i1 = i_node
                        j1 = j_node
                        k1 = k_node

                        # Nodes involved in the broken edges
                        n1 = current_solution[i1]
                        n2 = current_solution[(i1 + 1) % n]
                        n3 = current_solution[j1]
                        n4 = current_solution[(j1 + 1) % n]
                        n5 = current_solution[k1]
                        n6 = current_solution[(k1 + 1) % n]

                        old_cost = dm[n1][n2] + dm[n3][n4] + dm[n5][n6]
                        
                        # All 7 non-trivial 3-opt variants (some equivalent to 2-opt, some to other 3-opt forms)
                        # Here, we focus on the main permutations/reversals.

                        # Variant 1: A-C'-B'-D (segments: (i1+1..j1), (j1+1..k1), (k1+1..i1)) -> (i1+1..k1, reversed), (j1+1..i1, reversed)
                        # Reconnect (n1, n5), (n4, n2), (n3, n6)
                        new_cost_v1 = dm[n1][n5] + dm[n4][n2] + dm[n3][n6]
                        improvement_v1 = old_cost - new_cost_v1
                        
                        if improvement_v1 > best_improvement_found_in_pass:
                            best_improvement_found_in_pass = improvement_v1
                            best_i, best_j, best_k = i1, j1, k1
                            best_variant = 1

                        # Variant 2: A-B'-C'-D (segments: (i1+1..j1), (j1+1..k1), (k1+1..i1)) -> (i1+1..j1, reversed), (k1+1..j1, reversed)
                        # Reconnect (n1, n3), (n2, n5), (n4, n6)
                        new_cost_v2 = dm[n1][n3] + dm[n2][n5] + dm[n4][n6]
                        improvement_v2 = old_cost - new_cost_v2
                        
                        if improvement_v2 > best_improvement_found_in_pass:
                            best_improvement_found_in_pass = improvement_v2
                            best_i, best_j, best_k = i1, j1, k1
                            best_variant = 2

                        # Variant 3: A-C-B-D (segments: (i1+1..j1), (j1+1..k1), (k1+1..i1)) -> swap middle and third segments
                        # Reconnect (n1, n4), (n5, n2), (n3, n6)
                        new_cost_v3 = dm[n1][n4] + dm[n5][n2] + dm[n3][n6]
                        improvement_v3 = old_cost - new_cost_v3

                        if improvement_v3 > best_improvement_found_in_pass:
                            best_improvement_found_in_pass = improvement_v3
                            best_i, best_j, best_k = i1, j1, k1
                            best_variant = 3
            
            if best_improvement_found_in_pass > 1e-9:
                # Apply the best found 3-opt move based on its variant
                # Extract segments considering circularity
                tour_copy = current_solution[:]
                
                # To handle circular tours for 3-opt, we can normalize the tour by rotating
                # it so that best_i is at index 0, then applying regular slicing.
                # However, for 3-opt where we're just reversing/swapping defined segments,
                # directly constructing the new tour by slicing and concatenating parts is more robust.
                
                # General approach for 3-opt application (segments defined by i, j, k break points):
                # We break edges (i, i+1), (j, j+1), (k, k+1).
                # Segments are P1: tour[0...i], P2: tour[i+1...j], P3: tour[j+1...k], P4: tour[k+1...n-1]
                # The provided indices i, j, k define the *first node* of the *removed edge*.
                # So segments are [0...i], [i+1...j], [j+1...k], [k+1...N-1].
                # This assumes i < j < k for non-wrapping indices.
                
                # The current _best_improvement_3_opt_extended implementation already handles this assuming i,j,k are sorted
                # from a range that avoids wrap-around issues (n-3, n-2, n-1). This needs care if iterating over all N.
                # For this specific implementation, it correctly takes `i_node, j_node, k_node` as indices, not segments.
                # The segments are implicitly defined by (i_node, i_node+1), (j_node, j_node+1), (k_node, k_node+1).
                
                # Re-extract parts based on the best_i, best_j, best_k found
                part1 = current_solution[:best_i + 1]
                part2 = current_solution[best_i + 1 : best_j + 1]
                part3 = current_solution[best_j + 1 : best_k + 1]
                part4 = current_solution[best_k + 1 :]

                if best_variant == 1: # A-C'-B'-D
                    current_solution = part1 + part3[::-1] + part2[::-1] + part4
                elif best_variant == 2: # A-B'-C'-D
                    current_solution = part1 + part2[::-1] + part3[::-1] + part4
                elif best_variant == 3: # A-C-B-D
                    current_solution = part1 + part3 + part2 + part4
                else: 
                    break # Should not happen if best_variant is correctly set
            else:
                break
        return current_solution


    # --- Perturbation Operators ---

    def _random_k_opt_perturbation(solution: List[int], k_swaps: int) -> List[int]:
        """Applies 'k_swaps' random 2-opt swaps to perturb the solution."""
        perturbed_solution = solution[:]
        n = len(solution)
        if n < 2: return perturbed_solution

        for _ in range(k_swaps):
            if n < 2: break 
            if n == 2:
                idx1, idx2 = 0, 1
            else:
                idx_a, idx_b = random.sample(range(n), 2)
                idx1 = min(idx_a, idx_b)
                idx2 = max(idx_a, idx_b)
            
            perturbed_solution = _two_opt_swap(perturbed_solution, idx1, idx2)
        return perturbed_solution

    def _or_opt_segment_relocation_perturbation(solution: List[int]) -> List[int]:
        """Performs a single random Or-opt style segment relocation as a perturbation."""
        n = len(solution)
        if n < 3: return solution[:]
        
        segment_length = random.randint(1, min(3, n - 1))
        
        segment_start_idx = random.randint(0, n - 1)
        segment_end_inclusive_idx = (segment_start_idx + segment_length - 1) % n

        segment_to_move = []
        if segment_start_idx <= segment_end_inclusive_idx:
            segment_to_move = solution[segment_start_idx : segment_end_inclusive_idx + 1]
        else: # Segment wraps around
            segment_to_move = solution[segment_start_idx:] + solution[:segment_end_inclusive_idx + 1]
        
        remaining_tour = []
        if segment_start_idx <= segment_end_inclusive_idx:
            remaining_tour = solution[:segment_start_idx] + solution[segment_end_inclusive_idx + 1:]
        else:
            remaining_tour = solution[segment_end_inclusive_idx + 1 : segment_start_idx]
        
        insert_idx_in_remaining_tour = random.randint(0, len(remaining_tour))
        
        new_tour_candidate = remaining_tour[:insert_idx_in_remaining_tour] + segment_to_move + remaining_tour[insert_idx_in_remaining_tour:]
        
        return new_tour_candidate

    def _double_bridge_kick(solution: List[int]) -> List[int]:
        """
        Applies a double-bridge perturbation (4-opt move) for strong exploration.
        Falls back to a 2-opt perturbation for small N where double-bridge is not applicable.
        """
        n = len(solution)
        if n < 8: # Double bridge needs at least 8 nodes to ensure distinct segments
            return _random_k_opt_perturbation(solution, k_swaps=2)

        indices = random.sample(range(n), 4)
        indices.sort()
        
        i1, i2, i3, i4 = indices[0], indices[1], indices[2], indices[3]

        # Break points define segments: A (0..i1-1), B (i1..i2-1), C (i2..i3-1), D (i3..i4-1), E (i4..n-1)
        # New tour: A - D - C - B - E
        # Example: 0-1-2-3-4-5-6-7. Break (0,1), (2,3), (4,5), (6,7). 
        # Segments are (0), (1,2), (3,4), (5,6), (7).
        # Should be: tour[:i1], tour[i1:i2], tour[i2:i3], tour[i3:i4], tour[i4:]
        segment_A = solution[:i1]
        segment_B = solution[i1:i2]
        segment_C = solution[i2:i3]
        segment_D = solution[i3:i4]
        segment_E = solution[i4:]

        new_tour_candidate = segment_A + segment_D + segment_C + segment_B + segment_E
        
        if len(new_tour_candidate) != n or len(set(new_tour_candidate)) != n:
            return _random_k_opt_perturbation(solution, k_swaps=2)

        return new_tour_candidate

    def _random_3_opt_perturbation(solution: List[int]) -> List[int]:
        """
        Performs a single random 3-opt move by selecting one of the three standard 3-opt variants
        as a perturbation. Randomly chooses from the 3 variants for increased diversity.
        """
        n = len(solution)
        if n < 6: # Ensure enough cities for meaningful 3-opt perturbation
            return _random_k_opt_perturbation(solution, k_swaps=1)

        # Select three distinct break points (indices of starting nodes of edges) for 3-opt
        # Ensure points are distinct and far enough apart for meaningful segments.
        attempts = 0
        max_attempts = 100
        found_valid_indices = False
        i, j, k = -1, -1, -1

        while attempts < max_attempts and not found_valid_indices:
            # Select three distinct indices that will be the first node of the cut edge
            p = random.sample(range(n), 3)
            p.sort()
            test_i, test_j, test_k = p[0], p[1], p[2]

            # Ensure segments between break points are not empty (e.g., j-i > 0)
            # This is implicitly handled by random.sample and sort, but good to be careful if indices are near `n`
            if (test_j - test_i > 0 and test_k - test_j > 0 and 
                (n - 1 - test_k + test_i + 1) > 0): # Check for circular segment length
                i, j, k = test_i, test_j, test_k
                found_valid_indices = True
            attempts += 1
        
        if not found_valid_indices: # Fallback if suitable indices not found (highly unlikely for n >= 6)
            return _random_k_opt_perturbation(solution, k_swaps=1)

        # Segments based on chosen indices for edges (i, i+1), (j, j+1), (k, k+1)
        part1 = solution[:i + 1]
        part2 = solution[i + 1 : j + 1]
        part3 = solution[j + 1 : k + 1]
        part4 = solution[k + 1 :]

        # Randomly choose one of the three 3-opt variants
        variant_type = random.randint(1, 3) 
        perturbed_tour = []

        if variant_type == 1: # A-C'-B'-D
            perturbed_tour = part1 + part3[::-1] + part2[::-1] + part4
        elif variant_type == 2: # A-B'-C'-D
            perturbed_tour = part1 + part2[::-1] + part3[::-1] + part4
        elif variant_type == 3: # A-C-B-D
            perturbed_tour = part1 + part3 + part2 + part4
        
        if len(perturbed_tour) != n or len(set(perturbed_tour)) != n:
            return _random_k_opt_perturbation(solution, k_swaps=1)

        return perturbed_tour

    def _segment_reverse_relocate_perturbation(solution: List[int]) -> List[int]:
        """
        Performs a segment reversal followed by a relocation as a perturbation,
        combining 2-opt and Or-opt ideas for a stronger kick.
        """
        n = len(solution)
        if n < 3: return solution[:]

        segment_length = random.randint(1, max(2, n // 4)) # Max length up to N/4, minimum 2
        segment_start_idx = random.randint(0, n - 1)
        segment_end_inclusive_idx = (segment_start_idx + segment_length - 1) % n

        segment_to_move = []
        if segment_start_idx <= segment_end_inclusive_idx:
            segment_to_move = solution[segment_start_idx : segment_end_inclusive_idx + 1]
        else: # Segment wraps around
            segment_to_move = solution[segment_start_idx:] + solution[:segment_end_inclusive_idx + 1]
        
        # Reverse the segment
        segment_to_move.reverse()

        # Create the remaining tour after segment removal
        remaining_tour = []
        if segment_start_idx <= segment_end_inclusive_idx:
            remaining_tour = solution[:segment_start_idx] + solution[segment_end_inclusive_idx + 1:]
        else: 
            remaining_tour = solution[segment_end_inclusive_idx + 1 : segment_start_idx]
        
        insert_idx_in_remaining_tour = random.randint(0, len(remaining_tour))
        
        new_tour_candidate = remaining_tour[:insert_idx_in_remaining_tour] + segment_to_move + remaining_tour[insert_idx_in_remaining_tour:]
        
        if len(new_tour_candidate) != n or len(set(new_tour_candidate)) != n:
             return solution[:] # Return original if something went wrong

        return new_tour_candidate


    # --- Variable Neighborhood Descent (VND) implementation ---

    def _variable_neighborhood_descent(initial_solution: List[int], dm: List[List[float]]) -> List[int]:
        """
        Applies VND to find a local optimum by iteratively exploring neighborhoods.
        Sequence: 2-opt -> Or-opt (1-3) -> Extended 3-opt.
        """
        current_solution = initial_solution[:]
        
        neighborhood_operators_sequence = [
            _best_improvement_2_opt, 
            _best_improvement_or_opt,
            _best_improvement_3_opt_extended 
        ]
        
        k = 0 
        while k < len(neighborhood_operators_sequence):
            operator = neighborhood_operators_sequence[k]
            
            improved_solution = operator(current_solution, dm)
            
            current_cost = _calculate_tour_distance(current_solution, dm)
            improved_cost = _calculate_tour_distance(improved_solution, dm)

            if improved_cost < current_cost - 1e-9: 
                current_solution = improved_solution
                k = 0 
            else:
                k += 1 
        return current_solution

    # --- Main ILS Mutation Logic ---

    # Adaptive configuration parameters based on problem size (n)
    MAX_ILS_ITERATIONS: int
    INITIAL_TEMP: float
    COOLING_RATE: float
    base_perturbation_k_factor: float # Adjusted to be a factor for random_k_opt

    if number_of_city <= 50:
        MAX_ILS_ITERATIONS = 75 
        INITIAL_TEMP = 35.0 
        COOLING_RATE = 0.975 
        base_perturbation_k_factor = 0.12 
    elif number_of_city <= 200:
        MAX_ILS_ITERATIONS = 50
        INITIAL_TEMP = 25.0
        COOLING_RATE = 0.98
        base_perturbation_k_factor = 0.08
    else: # number_of_city > 200
        MAX_ILS_ITERATIONS = 30 
        INITIAL_TEMP = 15.0
        COOLING_RATE = 0.985
        base_perturbation_k_factor = 0.06 
    
    # Adaptive Operator Selection (AOS) for perturbation operators
    perturbation_operators = [
        _random_k_opt_perturbation, 
        _or_opt_segment_relocation_perturbation, 
        _double_bridge_kick,
        _random_3_opt_perturbation,
        _segment_reverse_relocate_perturbation # Added from v1
    ]
    perturb_operator_scores = {op.__name__: 1.0 for op in perturbation_operators}
    PERTURB_AOS_ALPHA = 0.15 

    # Initialize ILS: first, find a local optimum from the child tour
    current_tour_ils_state = _variable_neighborhood_descent(child[:], distance_matrix)
    best_overall_tour = current_tour_ils_state[:]
    best_overall_distance = _calculate_tour_distance(best_overall_tour, distance_matrix)
    
    current_temp = INITIAL_TEMP

    # Stagnation handling parameters (from v0, more nuanced)
    no_improvement_count = 0
    stagnation_threshold_mild = max(5, MAX_ILS_ITERATIONS // 8) 
    stagnation_threshold_strong = max(10, MAX_ILS_ITERATIONS // 4) 
    stagnation_perturb_increase_factor = 1.5 
    stagnation_temp_reheat_factor = 0.75 
    stagnation_threshold_extreme = max(15, MAX_ILS_ITERATIONS // 2) # For a more aggressive kick

    # Main ILS loop: Perturb, Local Search, and Accept iteratively
    for iteration in range(MAX_ILS_ITERATIONS):
        # Adaptive Perturbation Selection
        total_perturb_score = sum(perturb_operator_scores.values())
        if total_perturb_score < 1e-9: # Prevent division by zero if all scores are too low
            perturb_operator_scores.update({name: 1.0 for name in perturb_operator_scores})
            total_perturb_score = sum(perturb_operator_scores.values())
            
        operator_probs = [perturb_operator_scores[op.__name__] / total_perturb_score for op in perturbation_operators]
        chosen_perturbation_op = random.choices(perturbation_operators, weights=operator_probs, k=1)[0]

        # Determine perturbation intensity: adaptively increase if stagnating
        current_perturb_k_factor_adjusted = base_perturbation_k_factor
        
        # Apply more aggressive stagnation responses
        if no_improvement_count >= stagnation_threshold_extreme:
            current_perturb_k_factor_adjusted = max(0.20, base_perturbation_k_factor * 2.0) # Larger kick
            chosen_perturbation_op = _double_bridge_kick # Force a strong, high-diversity kick
        elif no_improvement_count >= stagnation_threshold_strong:
            current_perturb_k_factor_adjusted *= stagnation_perturb_increase_factor 
            current_temp = max(current_temp, INITIAL_TEMP * stagnation_temp_reheat_factor) # Reheat temp
            no_improvement_count = 0 # Reset count after strong action
        elif no_improvement_count >= stagnation_threshold_mild:
            current_perturb_k_factor_adjusted *= stagnation_perturb_increase_factor 
            
        # Apply the chosen perturbation operator
        perturbed_tour = []
        if chosen_perturbation_op == _random_k_opt_perturbation:
            # Crucial adaptive cap on k-swaps based on sqrt(N) for better scalability
            max_k_swaps_sqrt_n_cap = max(1, int(0.3 * math.sqrt(number_of_city))) 
            actual_k_swaps = max(1, min(int(number_of_city * current_perturb_k_factor_adjusted), max_k_swaps_sqrt_n_cap))
            perturbed_tour = chosen_perturbation_op(current_tour_ils_state, actual_k_swaps)
        else:
            perturbed_tour = chosen_perturbation_op(current_tour_ils_state)
        
        # Re-optimize the perturbed tour using VND
        candidate_tour = _variable_neighborhood_descent(perturbed_tour, distance_matrix)
        
        # Acceptance Criterion: Simulated Annealing-like
        current_ils_state_distance = _calculate_tour_distance(current_tour_ils_state, distance_matrix)
        candidate_distance = _calculate_tour_distance(candidate_tour, distance_matrix)

        improvement_reward = 0.0 

        if candidate_distance < best_overall_distance - 1e-9: # Significant overall improvement
            # Reward higher for new best global solution
            reward_factor = (best_overall_distance - candidate_distance) / best_overall_distance if best_overall_distance > 1e-9 else 0.0
            improvement_reward = 1.0 + reward_factor * 0.5 
            
            best_overall_tour = candidate_tour[:]
            best_overall_distance = candidate_distance
            current_tour_ils_state = candidate_tour[:] 
            no_improvement_count = 0 
            
        elif candidate_distance < current_ils_state_distance - 1e-9: # Improvement over current ILS state (but not global best)
            reward_factor = (current_ils_state_distance - candidate_distance) / current_ils_state_distance if current_ils_state_distance > 1e-9 else 0.0
            improvement_reward = 0.5 + reward_factor * 0.5 

            current_tour_ils_state = candidate_tour[:]
            no_improvement_count = 0 
        else: # No improvement or worse, apply SA acceptance logic
            no_improvement_count += 1 
            if current_temp > 1e-9:
                delta_cost = candidate_distance - current_ils_state_distance 
                acceptance_prob = math.exp(-delta_cost / current_temp)
                
                if random.random() < acceptance_prob:
                    current_tour_ils_state = candidate_tour[:]
                    improvement_reward = 0.1 # Small reward for accepting a worse solution to escape local optima
            
        # Update perturbation operator scores using an adaptive learning rule
        current_op_name = chosen_perturbation_op.__name__
        perturb_operator_scores[current_op_name] = (1 - PERTURB_AOS_ALPHA) * perturb_operator_scores[current_op_name] + PERTURB_AOS_ALPHA * improvement_reward
        
        # Ensure operator scores don't drop too low, keeping them in contention for selection
        for op_name in perturb_operator_scores:
            if perturb_operator_scores[op_name] < 0.05: 
                perturb_operator_scores[op_name] = 0.05
        
        # Cool the temperature for Simulated Annealing
        current_temp *= COOLING_RATE
        if current_temp < 0.05: 
            current_temp = 0.05 
            
    return best_overall_tour