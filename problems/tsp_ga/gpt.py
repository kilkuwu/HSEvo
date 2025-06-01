import random
from typing import List

class TSPHeuristics:
    """
    A helper class encapsulating various TSP local search and perturbation operators.
    This structure improves code organization and reusability.
    """
    def __init__(self, number_of_city: int, distance_matrix: List[List[float]]):
        self.number_of_city = number_of_city
        self.distance_matrix = distance_matrix

    def objective_function(self, tour: List[int]) -> float:
        """Calculate the total distance of a tour (fitness function)."""
        if not tour: # Robustness check
            return float('inf')
        total_distance = 0.0
        for i in range(len(tour) - 1):
            total_distance += self.distance_matrix[tour[i]][tour[i + 1]]
        total_distance += self.distance_matrix[tour[-1]][tour[0]] # Return to start
        return total_distance

    def two_opt_swap(self, solution: List[int], i: int, j: int) -> List[int]:
        """Performs a 2-opt swap operation by reversing a segment of the tour."""
        new_solution = solution[:]
        if i > j: # Ensure i is always less than j
            i, j = j, i
        new_solution[i:j+1] = list(reversed(new_solution[i:j+1]))
        return new_solution
    
    def _adaptive_two_opt_local_search(self, solution: List[int], random_attempts_per_pass: int = None) -> List[int]:
        """
        Performs 2-opt local search. If random_attempts_per_pass is None, it does a full N^2 scan (hill-climb).
        Otherwise, it performs a fixed number of random attempts per pass.
        This function iterates until no further improvement is made in a pass.
        """
        best_solution = solution[:]
        best_distance = self.objective_function(solution)
        
        if self.number_of_city < 2: 
            return best_solution

        improved = True
        while improved:
            improved = False
            
            if random_attempts_per_pass is None: # Perform a full 2-opt hill climb pass
                for i in range(self.number_of_city - 1):
                    for j in range(i + 1, self.number_of_city):
                        temp_solution = self.two_opt_swap(best_solution, i, j)
                        temp_distance = self.objective_function(temp_solution)
                        
                        if temp_distance < best_distance:
                            best_distance = temp_distance
                            best_solution = temp_solution
                            improved = True
                            break # Found improvement, restart this pass from beginning
                    if improved:
                        break 
            else: # For larger problems, perform a fixed number of random 2-opt attempts per pass.
                num_attempts = random_attempts_per_pass 
                for _ in range(num_attempts):
                    # Ensure i < j
                    i, j = sorted(random.sample(range(self.number_of_city), 2))
                    temp_solution = self.two_opt_swap(best_solution, i, j)
                    temp_distance = self.objective_function(temp_solution)
                    
                    if temp_distance < best_distance:
                        best_distance = temp_distance
                        best_solution = temp_solution
                        improved = True
                        break # Found an improvement, restart the while loop (new pass)
            
            if not improved: # If no improvement in the current pass, stop.
                break
                
        return best_solution

    def or_opt_reinsert(self, solution: List[int], start_idx: int, end_idx: int, insert_idx: int) -> List[int]:
        """
        Performs an Or-opt reinsertion operation: moves a contiguous segment
        (from start_idx to end_idx) to a new position (after insert_idx).
        Handles wrap-around for insertion point.
        """
        n = len(solution)
        if not (0 <= start_idx <= end_idx < n):
            return solution[:] # Invalid segment

        segment = solution[start_idx : end_idx + 1]
        temp_solution = solution[:start_idx] + solution[end_idx+1:]
        
        # Adjust insert_idx if the segment was removed before the insertion point
        if insert_idx > end_idx:
            insert_idx -= (end_idx - start_idx + 1)
        
        # Ensure insert_point is valid for temp_solution's bounds
        insert_point = max(-1, min(len(temp_solution) - 1, insert_idx)) 
        
        new_solution = temp_solution[:insert_point+1] + segment + temp_solution[insert_point+1:]
        return new_solution

    def _apply_or_opt_hill_climb(self, solution: List[int], max_segment_len: int = 3) -> List[int]:
        """Applies Or-opt hill climbing for segments of specified maximum length."""
        improved = True
        best_solution = solution[:]
        best_distance = self.objective_function(solution)
        n = self.number_of_city

        while improved:
            improved = False
            for seg_len in range(1, max_segment_len + 1):
                if seg_len >= n: 
                    continue

                for i in range(n): 
                    start_idx = i
                    end_idx = (i + seg_len - 1) 
                    if end_idx >= n: # Skip wrapping segments for this deterministic hill climb
                        continue

                    for j in range(n): # Potential insertion index (after this city)
                        is_trivial_insert = False
                        
                        # Check for trivial moves: reinserting at original position or adjacent causing no change
                        if start_idx <= j <= end_idx: # Insertion point is within the segment being moved
                            is_trivial_insert = True
                        elif j == (start_idx - 1 + n) % n: # Inserting right before segment (cyclical)
                            is_trivial_insert = True
                        elif j == end_idx: # Inserting right after segment (cyclical)
                            is_trivial_insert = True

                        if not is_trivial_insert:
                            new_solution = self.or_opt_reinsert(best_solution, start_idx, end_idx, j)
                            new_distance = self.objective_function(new_solution)

                            if new_distance < best_distance:
                                best_distance = new_distance
                                best_solution = new_solution
                                improved = True
                                break 
                    if improved:
                        break
                if improved:
                    break
        return best_solution

    def double_bridge_swap(self, solution: List[int]) -> List[int]:
        """
        Performs a double bridge (4-opt) swap. Breaks 4 edges and reconnects them
        to escape local optima. Requires at least 8 cities for proper operation.
        """
        n = len(solution)
        if n < 8:
            return solution[:]

        p1, p2, p3, p4 = sorted(random.sample(range(n), 4))
        
        new_solution = (solution[0:p1] +           # segment_A
                        solution[p3:p4] +           # segment_D
                        solution[p2:p3] +           # segment_C
                        solution[p1:p2] +           # segment_B
                        solution[p4:n])             # segment_E
        
        return new_solution

    def _random_or_opt_perturbation(self, tour: List[int]) -> List[int]:
        """
        Performs a randomized Or-opt style perturbation:
        selects a random segment and reinserts it at a random valid position.
        The segment length is adaptively chosen.
        """
        n = len(tour)
        if n < 2: return tour[:]

        seg_len = random.randint(1, min(n // 3, 5)) 
        if seg_len == 0: seg_len = 1 # ensure at least length 1

        start_idx = random.randint(0, n - seg_len)
        end_idx = start_idx + seg_len - 1
        
        valid_insert_indices = []
        for j in range(n):
            is_trivial = False
            # Check for trivial moves: inserting where it was, or adjacent causing no change
            if start_idx <= j <= end_idx: 
                is_trivial = True
            elif j == (start_idx - 1 + n) % n: 
                is_trivial = True
            elif j == end_idx: 
                is_trivial = True
            
            if not is_trivial:
                valid_insert_indices.append(j)

        if valid_insert_indices:
            insert_idx = random.choice(valid_insert_indices)
            return self.or_opt_reinsert(tour, start_idx, end_idx, insert_idx)
        else: # Fallback for very small N where few/no valid reinsertions exist
            # Perform a simple 2-opt swap as a minimal perturbation
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            return self.two_opt_swap(tour, i, j)

    def _random_segment_swap_perturbation(self, tour: List[int]) -> List[int]:
        """
        Performs a perturbation by swapping two random, non-overlapping segments of the tour.
        This is a type of 4-opt move (breaks 4 edges).
        """
        n = len(tour)
        if n < 6: # Need at least 6 cities to form two distinct non-overlapping segments and remaining parts
            return tour[:]
        
        # Choose 4 distinct points (indices) as cut points (0 <= p0 < p1 < p2 < p3 < n)
        p0, p1, p2, p3 = sorted(random.sample(range(n), 4))

        segment_A = tour[0:p0]
        segment_S1 = tour[p0:p1]
        segment_M = tour[p1:p2] # Middle segment
        segment_S2 = tour[p2:p3]
        segment_E = tour[p3:n]

        # Reconnection pattern: A - S2 - M - S1 - E
        new_tour = segment_A + segment_S2 + segment_M + segment_S1 + segment_E
        return new_tour
    
    def three_opt_random_swap_variant(self, solution: List[int]) -> List[int]:
        """
        Applies a random 3-opt specific variant: breaks 3 edges and reconnects them.
        This particular variant swaps two segments (B and C) and reverses one of them (C).
        Original: A-B-C-D. New: A-C_rev-B_rev-D.
        """
        n = len(solution)
        if n < 3: # Need at least 3 cities to break 3 edges (or 3 segments)
            return solution[:]
        
        # Choose 3 distinct indices as cut points
        p1, p2, p3 = sorted(random.sample(range(n), 3))
        
        segment_A = solution[:p1+1]       # Segment A (includes p1)
        segment_B = solution[p1+1:p2+1]   # Segment B (includes p2)
        segment_C = solution[p2+1:p3+1]   # Segment C (includes p3)
        segment_D = solution[p3+1:]       # Segment D (rest)

        # One common 3-opt rearrangement: A + reversed(C) + reversed(B) + D
        new_solution_candidate = segment_A + list(reversed(segment_C)) + list(reversed(segment_B)) + segment_D
        
        return new_solution_candidate

def mutate_v2(number_of_city: int, child: List[int], distance_matrix: List[List[float]]) -> List[int]:
    """
    Mutates a TSP tour (child) using an advanced Iterated Local Search (ILS) approach
    with adaptive perturbation strategies and robust re-optimization.
    """
    heuristics = TSPHeuristics(number_of_city, distance_matrix)
    
    current_tour = list(child) 
    best_overall_tour = list(child)
    best_overall_distance = heuristics.objective_function(child)

    # Phase 1: Initial Deep Local Search (Exploitation)
    # Start with a robust 2-opt local search. For smaller N, this will be exhaustive.
    # For larger N, it will do a significant number of random attempts.
    initial_2opt_attempts = None # None implies full scan for N <= 150 (in _adaptive_two_opt_local_search)
    if number_of_city > 150:
        initial_2opt_attempts = number_of_city * 2 # A substantial number of random attempts for larger N
    current_tour = heuristics._adaptive_two_opt_local_search(current_tour, random_attempts_per_pass=initial_2opt_attempts)
    
    # Conditionally apply Or-opt hill climbing for smaller/medium instances where it's feasible
    # This is an N^3 operation, so keep the N limit strict for performance.
    if number_of_city <= 100: 
        max_or_opt_seg_len_climb = min(3, number_of_city // 5) # Max segment length for Or-opt, prevent too large
        if max_or_opt_seg_len_climb == 0: max_or_opt_seg_len_climb = 1
        current_tour = heuristics._apply_or_opt_hill_climb(current_tour, max_segment_len=max_or_opt_seg_len_climb)
        # Polish with 2-opt after Or-opt, as Or-opt might create new 2-opt opportunities.
        current_tour = heuristics._adaptive_two_opt_local_search(current_tour, random_attempts_per_pass=initial_2opt_attempts) 
    
    best_overall_tour = list(current_tour)
    best_overall_distance = heuristics.objective_function(current_tour)

    # Phase 2: Iterated Local Search (Exploration & Refinement)
    
    # Adaptive number of perturbations based on problem size.
    max_perturb_iterations = 15 
    if number_of_city > 100:
        max_perturb_iterations = min(number_of_city // 8, 40) 
    
    no_improvement_count = 0
    # Dynamic threshold for no improvement: allows for more patience on larger instances.
    max_no_improvement_threshold = 7 
    if number_of_city > 150:
        max_no_improvement_threshold = min(number_of_city // 15, 12) 

    # Re-optimization intensity after perturbation
    # For smaller N, perform full 2-opt search (None). For larger N, use targeted attempts.
    re_opt_attempts_per_pass = max(10, number_of_city // 10) if number_of_city > 100 else None 

    for iteration in range(max_perturb_iterations):
        perturbed_tour = list(best_overall_tour) # Always perturb the current best solution found

        # --- Adaptive Perturbation Strategy ---
        # Operators grouped by general disruption level
        perturbation_ops_light = [heuristics._random_or_opt_perturbation]
        perturbation_ops_medium = [heuristics.three_opt_random_swap_variant, heuristics._random_segment_swap_perturbation]
        perturbation_ops_strong = [heuristics.double_bridge_swap]

        # Base probabilities for selecting a category of perturbation
        # Tendency to use lighter perturbations initially, boost stronger ones on stagnation
        prob_light = 0.4
        prob_medium = 0.3
        prob_strong = 0.3
        
        # Adjust probabilities based on stagnation
        if no_improvement_count >= max_no_improvement_threshold / 2: # Moderate stagnation
            prob_light = max(0.1, prob_light - 0.15)
            prob_medium = min(0.5, prob_medium + 0.05)
            prob_strong = min(0.6, prob_strong + 0.10)
        
        if no_improvement_count >= max_no_improvement_threshold * 0.8: # High stagnation
            prob_light = max(0.05, prob_light - 0.2)
            prob_medium = min(0.6, prob_medium + 0.1)
            prob_strong = min(0.7, prob_strong + 0.1)

        # Normalize probabilities to sum to 1.0
        total_prob = prob_light + prob_medium + prob_strong
        prob_light /= total_prob
        prob_medium /= total_prob
        prob_strong /= total_prob

        # Choose a category, then an operator from that category
        chosen_category = random.choices([perturbation_ops_light, perturbation_ops_medium, perturbation_ops_strong],
                                         weights=[prob_light, prob_medium, prob_strong], k=1)[0]
        
        chosen_operator = random.choice(chosen_category)
        
        # Apply the chosen perturbation
        perturbed_tour = chosen_operator(perturbed_tour)

        # A "kick" strategy for very high stagnation: apply multiple perturbations
        # This provides an additional diversification step when the search is stuck
        if no_improvement_count >= max_no_improvement_threshold:
            if random.random() < 0.6: # 60% chance for an extra light kick
                extra_operator = random.choice(perturbation_ops_light)
                perturbed_tour = extra_operator(perturbed_tour)
            if random.random() < 0.3 and number_of_city >= 6: # 30% chance for an extra medium kick
                extra_operator = random.choice(perturbation_ops_medium)
                perturbed_tour = extra_operator(perturbed_tour)

        # --- Re-optimize after perturbation with adaptive 2-opt local search ---
        perturbed_tour = heuristics._adaptive_two_opt_local_search(perturbed_tour, random_attempts_per_pass=re_opt_attempts_per_pass)

        # --- Acceptance Criteria: Greedy ---
        new_distance = heuristics.objective_function(perturbed_tour)
        if new_distance < best_overall_distance:
            best_overall_tour = list(perturbed_tour)
            best_overall_distance = new_distance
            no_improvement_count = 0 # Reset counter on improvement
        else:
            no_improvement_count += 1
        
        # Adaptive termination: Stop ILS loop if no improvement for too long
        if no_improvement_count >= max_no_improvement_threshold:
            break 

    return best_overall_tour