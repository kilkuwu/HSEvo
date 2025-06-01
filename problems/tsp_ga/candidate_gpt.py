import random
import math
from typing import List, Dict, Callable

def mutate(number_of_city: int, child: List[int], distance_matrix: List[List[float]]) -> List[int]:
    """
    Applies an advanced Iterated Local Search (ILS) with adaptive Variable Neighborhood Descent (VND)
    and an adaptive operator selection mechanism to robustly improve a TSP tour.

    Key idea: Combines efficient O(1) delta local searches (2-opt, Or-opt, 3-opt)
    with diverse, adaptively chosen perturbations and SA-like acceptance, scaling
    search intensity based on problem size and stagnation.
    """

    # --- Helper Functions (Optimized for O(1) delta calculations) ---

    def objective_function(tour: List[int], distance_matrix: List[List[float]]) -> float:
        """Calculate the total distance of a tour."""
        total_distance = 0.0
        num_cities = len(tour)
        if num_cities < 2:
            return 0.0
        for i in range(num_cities - 1):
            total_distance += distance_matrix[tour[i]][tour[i + 1]]
        total_distance += distance_matrix[tour[-1]][tour[0]]
        return total_distance

    def two_opt_swap(solution: List[int], i: int, j: int) -> List[int]:
        """Performs a 2-opt swap operation by reversing a segment of the tour."""
        new_solution = solution[:]
        if i > j:
            i, j = j, i
        new_solution[i:j+1] = list(reversed(new_solution[i:j+1]))
        return new_solution

    def _get_distance_delta_2opt(current_tour: List[int], i: int, j: int, num_cities: int, distance_matrix: List[List[float]]) -> float:
        """
        Calculates the change in tour distance for a 2-opt swap (i,j) in O(1).
        Returns new_cost - old_cost. Negative implies improvement.
        """
        p_i_node = current_tour[(i - 1 + num_cities) % num_cities]
        c_i_node = current_tour[i]
        p_j_node = current_tour[j]
        c_j_node = current_tour[(j + 1) % num_cities]

        cost_removed = distance_matrix[p_i_node][c_i_node] + \
                       distance_matrix[p_j_node][c_j_node]
        
        cost_added = distance_matrix[p_i_node][p_j_node] + \
                     distance_matrix[c_i_node][c_j_node]
        
        return cost_added - cost_removed

    def two_opt_local_search(tour: List[int], distance_matrix: List[List[float]], 
                             steepest_descent: bool = False, max_passes: int = -1) -> List[int]:
        """
        Performs a 2-opt hill climbing local search (first-improvement or steepest descent).
        Optimized with O(1) delta calculations per move, resulting in O(N^2) complexity per pass.
        """
        num_cities = len(tour)
        if num_cities < 2: return tour[:]

        best_solution = tour[:]
        
        passes_count = 0
        while max_passes == -1 or passes_count < max_passes:
            passes_count += 1
            improved_in_this_pass = False
            
            if steepest_descent:
                best_i, best_j = -1, -1
                best_improvement_delta = 0.0 # Looking for negative delta
                
                for i in range(num_cities):
                    for j in range(i + 2, num_cities): # Ensures distinct edges and valid segment to reverse
                        delta = _get_distance_delta_2opt(best_solution, i, j, num_cities, distance_matrix)
                        
                        if delta < best_improvement_delta:
                            best_improvement_delta = delta
                            best_i, best_j = i, j
                            improved_in_this_pass = True
                
                if improved_in_this_pass and best_improvement_delta < -1e-9: # Significant improvement
                    best_solution = two_opt_swap(best_solution, best_i, best_j)
                else:
                    break # No significant improvement in this pass
            else: # First-improvement
                found_improvement_in_loop = False
                for i in range(num_cities):
                    for j in range(i + 2, num_cities):
                        delta = _get_distance_delta_2opt(best_solution, i, j, num_cities, distance_matrix)
                        
                        if delta < -1e-9: # Significant improvement found
                            best_solution = two_opt_swap(best_solution, i, j)
                            improved_in_this_pass = True
                            found_improvement_in_loop = True
                            break # Apply first improvement and restart pass
                    if found_improvement_in_loop:
                        break
            
            if not improved_in_this_pass:
                break # No improvement in this pass, terminate local search
            
        return best_solution

    def _get_distance_delta_or_opt_v2(current_tour: List[int], seg_start_idx: int, seg_len: int, target_insert_idx: int, num_cities: int, distance_matrix: List[List[float]]) -> float:
        """
        Calculates the change in tour distance for relocating a segment in O(1).
        Returns delta (new_cost - old_cost). Negative is improvement.
        Returns 0.0 if the move is invalid or trivial.
        """
        if num_cities < 2 or seg_len <= 0 or seg_len >= num_cities:
            return 0.0

        # Indices of nodes involved in the original connections
        idx_A = (seg_start_idx - 1 + num_cities) % num_cities # Node before segment start
        idx_B = seg_start_idx                                # Segment start node
        idx_C = (seg_start_idx + seg_len - 1) % num_cities   # Segment end node
        idx_D = (seg_start_idx + seg_len) % num_cities       # Node after segment end
        
        # Indices of nodes involved in the target insertion point
        idx_X = target_insert_idx                            # Node before insertion point
        idx_Y = (target_insert_idx + 1) % num_cities         # Node after insertion point
        
        node_A = current_tour[idx_A]
        node_B = current_tour[idx_B]
        node_C = current_tour[idx_C]
        node_D = current_tour[idx_D]
        node_X = current_tour[idx_X]
        node_Y = current_tour[idx_Y]

        # Helper to check if an index falls within a circular segment
        def is_idx_in_segment_range_circular(k_idx, s_idx, e_idx, n_cities):
            if s_idx <= e_idx: return s_idx <= k_idx <= e_idx
            else: return k_idx >= s_idx or k_idx <= e_idx

        # Check for invalid or trivial moves (e.g., inserting segment back into its original spot)
        # This prevents breaking non-existent edges or inserting adjacent to itself
        if (idx_X == idx_C and idx_Y == idx_D) or \
           (idx_X == idx_A and idx_Y == idx_B) or \
           is_idx_in_segment_range_circular(idx_X, idx_B, idx_C, num_cities) or \
           is_idx_in_segment_range_circular(idx_Y, idx_B, idx_C, num_cities):
            return 0.0

        cost_removed = distance_matrix[node_A][node_B] + \
                       distance_matrix[node_C][node_D] + \
                       distance_matrix[node_X][node_Y]
        
        cost_added = distance_matrix[node_A][node_D] + \
                     distance_matrix[node_X][node_B] + \
                     distance_matrix[node_C][node_Y]
        
        return cost_added - cost_removed

    def do_or_opt_move_v2(tour_arg: List[int], seg_start_idx_val: int, seg_len_val: int, target_insert_idx_val: int, num_cities_val: int) -> List[int]:
        """Performs the actual Or-opt move (segment relocation) in O(N)."""
        segment_to_move = [tour_arg[(seg_start_idx_val + k) % num_cities_val] for k in range(seg_len_val)]
        
        temp_remaining_tour = []
        segment_indices_set = set((seg_start_idx_val + k) % num_cities_val for k in range(seg_len_val))
        
        for k_all in range(num_cities_val):
            if k_all not in segment_indices_set:
                temp_remaining_tour.append(tour_arg[k_all])

        insert_pos_in_remaining = 0
        if num_cities_val > 0:
            original_node_at_insert_idx = tour_arg[target_insert_idx_val]
            try:
                # Find the position where original_node_at_insert_idx is in the remaining tour
                # This will be the point *before* which we want to insert the segment
                insert_pos_in_remaining = temp_remaining_tour.index(original_node_at_insert_idx) + 1
            except ValueError:
                # If target_insert_idx was part of the removed segment, or it's the last node
                # after the removed segment, insert at the end of remaining tour
                insert_pos_in_remaining = len(temp_remaining_tour)
        
        new_tour = temp_remaining_tour[:insert_pos_in_remaining] + segment_to_move + temp_remaining_tour[insert_pos_in_remaining:]
        return new_tour

    def or_opt_local_search_optimized(tour: List[int], distance_matrix: List[List[float]], segment_lengths: List[int] = None) -> List[int]:
        """
        Performs an Or-opt local search (first-improvement) using O(1) delta calculation.
        Overall O(N^2) per pass (if single segment_length).
        """
        if segment_lengths is None: segment_lengths = [1, 2, 3]

        num_cities = len(tour)
        current_tour = tour[:]
        
        improved = True
        while improved:
            improved = False
            for seg_len in segment_lengths:
                if seg_len >= num_cities or num_cities < 2: continue
                # Iterate through all possible starting positions of the segment
                for seg_start_idx in range(num_cities):
                    # Iterate through all possible insertion points
                    for target_insert_idx in range(num_cities):
                        delta = _get_distance_delta_or_opt_v2(current_tour, seg_start_idx, seg_len, target_insert_idx, num_cities, distance_matrix)
                        if delta < -1e-9: # If improvement found
                            current_tour = do_or_opt_move_v2(current_tour, seg_start_idx, seg_len, target_insert_idx, num_cities)
                            improved = True
                            break # Found first improvement, restart search for this seg_len
                    if improved: break 
                if improved: break # Found improvement, restart the entire search (over all seg_lengths)
        return current_tour

    def _get_distance_delta_3opt(current_tour: List[int], i: int, j: int, k: int, num_cities: int, distance_matrix: List[List[float]], move_type: int) -> float:
        """
        Calculates the change in tour distance for one of the 7 3-opt move types in O(1).
        i, j, k are cut points (indices of nodes *before* the cut edge).
        Assumes 0 <= i < j < k < num_cities.
        """
        n1 = current_tour[i]
        n2 = current_tour[(i + 1) % num_cities]
        n3 = current_tour[j]
        n4 = current_tour[(j + 1) % num_cities]
        n5 = current_tour[k]
        n6 = current_tour[(k + 1) % num_cities]

        old_cost = distance_matrix[n1][n2] + distance_matrix[n3][n4] + distance_matrix[n5][n6]
        new_cost = 0.0

        # Based on standard 3-opt move types
        # Segments: A = tour[0:i+1], B = tour[i+1:j+1], C = tour[j+1:k+1], D = tour[k+1:num_cities]
        if move_type == 1: # A + rev(B) + C + D => (n1, n3), (n2, n4), (n5, n6)
            new_cost = distance_matrix[n1][n3] + distance_matrix[n2][n4] + distance_matrix[n5][n6]
        elif move_type == 2: # A + B + rev(C) + D => (n1, n2), (n3, n5), (n4, n6)
            new_cost = distance_matrix[n1][n2] + distance_matrix[n3][n5] + distance_matrix[n4][n6]
        elif move_type == 3: # A + rev(B) + rev(C) + D => (n1, n3), (n2, n5), (n4, n6)
            new_cost = distance_matrix[n1][n3] + distance_matrix[n2][n5] + distance_matrix[n4][n6]
        elif move_type == 4: # A + C + B + D => (n1, n4), (n5, n2), (n3, n6)
            new_cost = distance_matrix[n1][n4] + distance_matrix[n5][n2] + distance_matrix[n3][n6]
        elif move_type == 5: # A + C + rev(B) + D => (n1, n4), (n5, n3), (n2, n6)
            new_cost = distance_matrix[n1][n4] + distance_matrix[n5][n3] + distance_matrix[n2][n6]
        elif move_type == 6: # A + rev(C) + B + D => (n1, n5), (n4, n2), (n3, n6)
            new_cost = distance_matrix[n1][n5] + distance_matrix[n4][n2] + distance_matrix[n3][n6]
        elif move_type == 7: # A + rev(C) + rev(B) + D => (n1, n5), (n4, n3), (n2, n6)
            new_cost = distance_matrix[n1][n5] + distance_matrix[n4][n3] + distance_matrix[n2][n6]
        else:
            return 0.0 # Should not happen with valid move_type

        return new_cost - old_cost

    def do_3_opt_move(tour: List[int], i: int, j: int, k: int, num_cities: int, move_type: int) -> List[int]:
        """Performs one of the 7 3-opt moves given cut points i, j, k. O(N) operation."""
        # Split tour into 4 segments based on i, j, k
        segment_A = tour[0 : i + 1]
        segment_B = tour[i + 1 : j + 1]
        segment_C = tour[j + 1 : k + 1]
        segment_D = tour[k + 1 : num_cities]

        # Apply the chosen move type by reordering and/or reversing segments
        if move_type == 1: return segment_A + list(reversed(segment_B)) + segment_C + segment_D
        elif move_type == 2: return segment_A + segment_B + list(reversed(segment_C)) + segment_D
        elif move_type == 3: return segment_A + list(reversed(segment_B)) + list(reversed(segment_C)) + segment_D
        elif move_type == 4: return segment_A + segment_C + segment_B + segment_D
        elif move_type == 5: return segment_A + segment_C + list(reversed(segment_B)) + segment_D
        elif move_type == 6: return segment_A + list(reversed(segment_C)) + segment_B + segment_D
        elif move_type == 7: return segment_A + list(reversed(segment_C)) + list(reversed(segment_B)) + segment_D
        else: return tour[:] # Fallback, should not be reached

    def three_opt_local_search_optimized(tour: List[int], distance_matrix: List[List[float]], 
                                         steepest_descent: bool = False, max_passes: int = 1) -> List[int]:
        """
        Performs a 3-opt local search using O(1) delta calculation.
        Overall complexity is O(N^3) per pass.
        """
        num_cities = len(tour)
        if num_cities < 4: return tour[:] # Need at least 4 cities for 3-opt to be meaningful

        current_solution = tour[:]
        
        passes_count = 0
        while max_passes == -1 or passes_count < max_passes:
            passes_count += 1
            improved_in_this_pass = False
            
            best_i, best_j, best_k, best_move_type = -1, -1, -1, -1
            best_improvement_delta = 0.0 # Looking for negative delta (cost reduction)

            # Iterate through all combinations of three distinct cut points
            for i in range(num_cities):
                for j in range(i + 2, num_cities): # j must be at least i+2 to ensure a segment B exists
                    for k in range(j + 2, num_cities): # k must be at least j+2 to ensure a segment C exists
                        
                        for move_type in range(1, 8): # Try all 7 possible 3-opt moves
                            delta = _get_distance_delta_3opt(current_solution, i, j, k, num_cities, distance_matrix, move_type)
                            
                            if delta < -1e-9: # If an improvement is found
                                if steepest_descent:
                                    if delta < best_improvement_delta: # Store best improvement for steepest descent
                                        best_improvement_delta = delta
                                        best_i, best_j, best_k, best_move_type = i, j, k, move_type
                                        improved_in_this_pass = True
                                else: # First-improvement strategy: apply immediately
                                    current_solution = do_3_opt_move(current_solution, i, j, k, num_cities, move_type)
                                    improved_in_this_pass = True
                                    return current_solution # Return immediately after first improvement
            
            if steepest_descent and improved_in_this_pass:
                # Apply the best move found in this pass for steepest descent
                current_solution = do_3_opt_move(current_solution, best_i, best_j, best_k, num_cities, best_move_type)
            
            if not improved_in_this_pass:
                break # No improvement in this pass, terminate local search
                
        return current_solution


    # --- Perturbation Operators (Diversification) ---
    # These operators generate new solutions by slightly altering the current best tour.

    def perturb_swap_random(tour: List[int]) -> List[int]:
        """Swaps two randomly chosen cities (equivalent to a 2-opt perturbation with a random segment)."""
        num_cities = len(tour)
        if num_cities < 2: return tour[:]
        new_tour = tour[:]
        idx1, idx2 = random.sample(range(num_cities), 2)
        new_tour[idx1], new_tour[idx2] = new_tour[idx2], new_tour[idx1]
        return new_tour

    def perturb_insert_random(tour: List[int]) -> List[int]:
        """Moves a randomly chosen city to a random new position (1-opt perturbation)."""
        num_cities = len(tour)
        if num_cities < 2: return tour[:]
        new_tour = tour[:]
        idx_to_move = random.randrange(num_cities)
        city = new_tour.pop(idx_to_move)
        insert_pos = random.randrange(num_cities)
        new_tour.insert(insert_pos, city)
        return new_tour

    def perturb_reverse_segment(tour: List[int]) -> List[int]:
        """Reverses a randomly chosen segment (simple 2-opt like perturbation)."""
        num_cities = len(tour)
        if num_cities < 2: return tour[:]
        new_tour = tour[:]
        
        i = random.randrange(num_cities)
        j = random.randrange(num_cities)
        
        if i == j: # Ensure at least two points for a segment
            j = (j + 1) % num_cities
            if i == j: return tour[:] # Still invalid segment (e.g., N=1)

        if i > j: i, j = j, i # Ensure i <= j for slicing
        
        new_tour[i:j+1] = list(reversed(new_tour[i:j+1]))
        return new_tour

    def perturb_scramble_segment(tour: List[int], min_len_factor: float, max_len_factor: float) -> List[int]:
        """Scrambles a randomly chosen segment with length adaptive to N."""
        num_cities = len(tour)
        if num_cities < 4: return tour[:] 
        new_tour = tour[:]
        
        # Calculate segment length based on factors of N
        min_len = max(2, int(num_cities * min_len_factor))
        max_len = max(min_len + 1, int(num_cities * max_len_factor))
        
        segment_len = random.randint(min_len, max_len)
        segment_len = min(segment_len, num_cities) # Ensure length doesn't exceed total cities
        
        # Ensure there's enough space to pick a segment
        if num_cities - segment_len < 0: return tour[:]
            
        start = random.randrange(0, num_cities - segment_len + 1) # Start index of the segment
        end = start + segment_len
        
        segment = new_tour[start:end]
        random.shuffle(segment) # Scramble the segment
        new_tour[start:end] = segment
        return new_tour

    def perturb_relocate_segment(tour: List[int], max_segment_len_factor: float) -> List[int]:
        """Relocates a small randomly chosen segment (Or-opt like perturbation)."""
        num_cities = len(tour)
        if num_cities < 2: return tour[:]
        new_tour = tour[:]
        
        max_seg_len_val = max(1, int(num_cities * max_segment_len_factor))
        segment_length = random.randint(1, min(max_seg_len_val, num_cities - 1)) # Small segment
        
        if num_cities <= segment_length: return tour[:]

        segment_start = random.randint(0, num_cities - segment_length) # Start index of segment to move
        
        segment = new_tour[segment_start : segment_start + segment_length]
        temp_tour = new_tour[:segment_start] + new_tour[segment_start + segment_length:] # Remove segment
        
        insert_pos = random.randint(0, len(temp_tour)) # Random insertion point in remaining tour
        
        final_tour = temp_tour[:insert_pos] + segment + temp_tour[insert_pos:]
        return final_tour

    def perturb_double_bridge(tour: List[int]) -> List[int]:
        """Performs a double-bridge (4-opt) perturbation - a strong, non-local move."""
        n = len(tour)
        if n < 8: return tour[:] # Requires at least 8 cities for meaningful non-trivial segments

        # Select 4 distinct cut points and sort them
        p = random.sample(range(n), 4)
        p.sort() 
        
        i, j, k, l = p[0], p[1], p[2], p[3]

        # Divide tour into 5 segments
        segment_A = tour[0:i]
        segment_B = tour[i:j]
        segment_C = tour[j:k]
        segment_D = tour[k:l]
        segment_E = tour[l:n]
        
        # Reorder segments to form the double-bridge move
        new_tour = segment_A + segment_D + segment_C + segment_B + segment_E
        return new_tour


    class OperatorManager:
        """
        Manages selection and performance tracking for perturbation operators.
        Uses a decaying score system and rewards based on improvement magnitude
        to adaptively select the most effective operators.
        """
        def __init__(self, operators: Dict[str, Callable]):
            self.operators = operators
            self.scores = {name: 1.0 for name in operators.keys()} # Initialize all scores
            self.learning_rate = 0.1 # Base rate for updating scores
            self.decay_rate = 0.98   # Rate at which all scores decay per iteration

        def select_operator(self) -> (Callable, str):
            """Selects an operator based on its current score (roulette wheel selection)."""
            current_scores = [max(0.01, s) for s in self.scores.values()] # Ensure scores stay positive
            total_score = sum(current_scores)

            if total_score == 0: # Fallback if all scores are zero
                return random.choice(list(self.operators.values())), "fallback"
            
            selected_name = random.choices(
                list(self.operators.keys()), 
                weights=current_scores, 
                k=1
            )[0]
            return self.operators[selected_name], selected_name

        def update_score(self, operator_name: str, improvement_found: bool, improvement_magnitude: float = 0.0):
            """Updates the score of an operator based on performance."""
            # Decay all operator scores to prioritize recent performance
            for name in self.scores:
                self.scores[name] *= self.decay_rate 
            
            # Reward successful operators more, penalize unsuccessful ones less
            if improvement_found:
                # Reward is scaled by magnitude of improvement, preventing small improvements from dominating
                reward = 0.05 + self.learning_rate * (improvement_magnitude / (improvement_magnitude + 1e-6))
                self.scores[operator_name] += reward * 5.0 # Stronger reward for finding true improvements
            else:
                self.scores[operator_name] -= self.learning_rate * 0.5 # Small penalty for no improvement
                
            self.scores[operator_name] = max(0.01, self.scores[operator_name]) # Keep score above a minimum


    # --- Main mutate function logic (Adaptive Iterated Local Search) ---
    
    current_best_tour = child[:]
    current_best_distance = objective_function(current_best_tour, distance_matrix)

    if number_of_city < 2: return current_best_tour # Trivial case

    # Define problem size thresholds for adapting strategy
    N_SMALL_THRESHOLD = 80
    N_MEDIUM_THRESHOLD = 250

    # --- Initial Local Optimization (Aggressive VND) ---
    # Apply a powerful VND sequence to reach a strong initial local optimum.
    # The set of operators applied depends on the number of cities.
    initial_vnd_operators = []
    if number_of_city <= N_SMALL_THRESHOLD:
        initial_vnd_operators = [
            lambda t: two_opt_local_search(t, distance_matrix, steepest_descent=True, max_passes=-1), # Thorough 2-opt
            lambda t: or_opt_local_search_optimized(t, distance_matrix, segment_lengths=[1,2,3]),     # 1-opt, 2-opt, 3-opt
            lambda t: three_opt_local_search_optimized(t, distance_matrix, steepest_descent=True, max_passes=1) # One steepest 3-opt pass
        ]
    elif number_of_city <= N_MEDIUM_THRESHOLD:
        initial_vnd_operators = [
            lambda t: two_opt_local_search(t, distance_matrix, steepest_descent=True, max_passes=1),   # One steep 2-opt pass
            lambda t: two_opt_local_search(t, distance_matrix, steepest_descent=False, max_passes=-1), # Then until 2-opt local optimum
            lambda t: or_opt_local_search_optimized(t, distance_matrix, segment_lengths=[1]),         # 1-opt (node relocation)
        ]
        if number_of_city <= 150: # 3-opt is more costly, apply only for moderate N
            initial_vnd_operators.append(lambda t: three_opt_local_search_optimized(t, distance_matrix, steepest_descent=False, max_passes=1))
    else: # For very large N, stick to more efficient O(N^2) operators
        initial_vnd_operators = [
            lambda t: two_opt_local_search(t, distance_matrix, steepest_descent=False, max_passes=-1),
            lambda t: or_opt_local_search_optimized(t, distance_matrix, segment_lengths=[1])
        ]
    
    current_vnd_tour = current_best_tour[:]
    current_vnd_distance = objective_function(current_vnd_tour, distance_matrix)

    # Execute VND sequence: if improvement in a neighborhood, restart from first neighborhood
    op_idx = 0
    while op_idx < len(initial_vnd_operators):
        op_func = initial_vnd_operators[op_idx]
        next_vnd_tour = op_func(current_vnd_tour)
        next_vnd_distance = objective_function(next_vnd_tour, distance_matrix)
        
        if next_vnd_distance < current_vnd_distance - 1e-9: # Significant improvement
            current_vnd_tour = next_vnd_tour[:]
            current_vnd_distance = next_vnd_distance
            op_idx = 0 # Restart VND sequence from the beginning
        else:
            op_idx += 1 # Move to the next neighborhood structure
    
    current_best_tour = current_vnd_tour[:]
    current_best_distance = current_vnd_distance

    # --- Adaptive Iterated Local Search (AILS) Parameters ---
    # Max iterations and initial temperature adapted based on problem size.
    max_ils_iterations = 0
    if number_of_city <= N_SMALL_THRESHOLD: max_ils_iterations = 30
    elif number_of_city <= N_MEDIUM_THRESHOLD: max_ils_iterations = 20
    else: max_ils_iterations = 12

    initial_temp_factor = 0.005 if number_of_city > 100 else 0.015
    initial_temp = current_best_distance * initial_temp_factor # Initial temperature scales with tour length
    cooling_rate = 0.99

    # Initialize perturbation operators for adaptive selection
    perturbation_operators = {
        "swap_random": perturb_swap_random,
        "insert_random": perturb_insert_random,
        "reverse_segment": perturb_reverse_segment,
        "scramble_segment_light": lambda t: perturb_scramble_segment(t, 0.03, 0.08), # Smaller scramble
        "scramble_segment_medium": lambda t: perturb_scramble_segment(t, 0.08, 0.2), # Medium scramble
        "relocate_segment_light": lambda t: perturb_relocate_segment(t, 0.03),      # Smaller relocate
        "relocate_segment_medium": lambda t: perturb_relocate_segment(t, 0.08),     # Medium relocate
        "double_bridge": perturb_double_bridge # Strongest perturbation
    }
    op_manager = OperatorManager(perturbation_operators)

    stagnation_threshold = 7 # Number of iterations without strict improvement to trigger special actions
    stagnation_counter = 0

    overall_best_tour = current_best_tour[:]
    overall_best_distance = current_best_distance

    temp = initial_temp # Current temperature for Simulated Annealing acceptance

    for ils_iter in range(max_ils_iterations):
        # 1. Perturbation (Diversification)
        # Select perturbation operator adaptively
        perturb_func, op_name = op_manager.select_operator()
        perturbed_tour = perturb_func(current_best_tour)
        
        # Skip if perturbation didn't change the tour (e.g., small N, or unlucky random pick)
        if perturbed_tour == current_best_tour:
            op_manager.update_score(op_name, False, 0.0)
            stagnation_counter += 1
            # Aggressive reheating if very stuck and temperature is low
            if temp < 1e-6 and stagnation_counter > stagnation_threshold * 2:
                temp = initial_temp * (1.0 + random.random()) 
            continue

        # 2. Local Search Re-optimization (Intensification within ILS)
        # Apply a sequence of VND local searches to the perturbed tour.
        reoptimized_tour = perturbed_tour[:]
        current_vnd_tour_distance = objective_function(reoptimized_tour, distance_matrix) 
        
        vnd_operators_ils = [
            lambda t: two_opt_local_search(t, distance_matrix, steepest_descent=False, max_passes=-1), # Basic 2-opt
            lambda t: or_opt_local_search_optimized(t, distance_matrix, segment_lengths=[1]),          # 1-opt
        ]
        # Deeper local search operators are conditionally added based on N and stagnation
        if number_of_city <= N_MEDIUM_THRESHOLD:
            vnd_operators_ils.insert(1, lambda t: or_opt_local_search_optimized(t, distance_matrix, segment_lengths=[1,2]))
            if number_of_city <= N_SMALL_THRESHOLD or (stagnation_counter >= stagnation_threshold // 2 and number_of_city <= N_MEDIUM_THRESHOLD):
                vnd_operators_ils.insert(2, lambda t: or_opt_local_search_optimized(t, distance_matrix, segment_lengths=[1,2,3]))
        if (number_of_city <= N_MEDIUM_THRESHOLD and number_of_city >= 4) or \
           (stagnation_counter >= stagnation_threshold and number_of_city <= N_MEDIUM_THRESHOLD * 1.5):
            vnd_operators_ils.append(lambda t: three_opt_local_search_optimized(t, distance_matrix, steepest_descent=False, max_passes=1))

        # Execute VND sequence for re-optimization
        op_idx_ils = 0
        while op_idx_ils < len(vnd_operators_ils):
            op_func_ils = vnd_operators_ils[op_idx_ils]
            next_vnd_tour_ils = op_func_ils(reoptimized_tour)
            next_vnd_distance_ils = objective_function(next_vnd_tour_ils, distance_matrix)
            
            if next_vnd_distance_ils < current_vnd_tour_distance - 1e-9:
                reoptimized_tour = next_vnd_tour_ils[:]
                current_vnd_tour_distance = next_vnd_distance_ils
                op_idx_ils = 0 # Restart VND within ILS if improvement
            else:
                op_idx_ils += 1

        reoptimized_distance = current_vnd_tour_distance

        # 3. Acceptance Criteria (Simulated Annealing-like)
        delta = reoptimized_distance - current_best_distance # New cost - Old cost

        strict_improvement_made = False
        improvement_magnitude_for_aos = 0.0

        if delta < -1e-9: # Strict improvement found
            current_best_tour = reoptimized_tour[:]
            current_best_distance = reoptimized_distance
            strict_improvement_made = True
            improvement_magnitude_for_aos = abs(delta)
            stagnation_counter = 0 # Reset stagnation counter
        elif temp > 1e-9: # Accept worse solution with probability (SA)
            acceptance_probability = math.exp(-delta / temp)
            if random.random() < acceptance_probability:
                current_best_tour = reoptimized_tour[:]
                current_best_distance = reoptimized_distance
                # Stagnation counter is NOT reset, as this isn't a strict step towards better optimum
            else:
                stagnation_counter += 1 # Solution not accepted
        else: # Temp is very low, only strict improvements are accepted
            stagnation_counter += 1

        # Update overall best tour found so far
        if current_best_distance < overall_best_distance:
            overall_best_tour = current_best_tour[:]
            overall_best_distance = current_best_distance

        # Update the selected perturbation operator's score
        op_manager.update_score(op_name, strict_improvement_made, improvement_magnitude_for_aos)
        
        # Adaptive strategy: React to stagnation
        if stagnation_counter >= stagnation_threshold:
            # Boost scores of strong perturbation operators to encourage larger jumps
            for name in ["double_bridge", "scramble_segment_medium"]:
                if name in op_manager.scores:
                    op_manager.scores[name] = max(op_manager.scores[name] * 3, 5.0) 
            # Reheat temperature to escape current local optima
            temp = initial_temp * (1.0 + random.random() * 0.8) 
            stagnation_counter = 0 # Reset stagnation counter after taking action

        # Cool down temperature
        temp *= cooling_rate
        if temp < 1e-12: temp = 1e-12 # Prevent temperature from becoming too small
            
    # Final Local Search: Ensure the returned tour is a strong local optimum before returning.
    # This acts as a final intensification phase.
    final_ls_tour = overall_best_tour[:]
    if number_of_city <= N_SMALL_THRESHOLD:
        final_ls_tour = two_opt_local_search(final_ls_tour, distance_matrix, steepest_descent=True, max_passes=-1)
        final_ls_tour = or_opt_local_search_optimized(final_ls_tour, distance_matrix, segment_lengths=[1,2,3])
        final_ls_tour = three_opt_local_search_optimized(final_ls_tour, distance_matrix, steepest_descent=True, max_passes=1)
    elif number_of_city <= N_MEDIUM_THRESHOLD:
        final_ls_tour = two_opt_local_search(final_ls_tour, distance_matrix, steepest_descent=False, max_passes=-1)
        final_ls_tour = or_opt_local_search_optimized(final_ls_tour, distance_matrix, segment_lengths=[1])
        if number_of_city <= 150:
            final_ls_tour = three_opt_local_search_optimized(final_ls_tour, distance_matrix, steepest_descent=False, max_passes=1)
    else: # For very large N, stick to efficient O(N^2) LS
        final_ls_tour = two_opt_local_search(final_ls_tour, distance_matrix, steepest_descent=False, max_passes=-1)

    return final_ls_tour