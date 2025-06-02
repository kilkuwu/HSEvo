import random
from typing import List


def mutate_v2(
    number_of_city: int, child: List[int], distance_matrix: List[List[float]]
) -> List[int]:
    """
    Applies an adaptive Iterated Local Search (ILS) mutation for TSP,
    incorporating refined adaptive strategies for local search operators
    (2-opt, Or-opt) and perturbation techniques, focusing on computational
    efficiency and balancing exploration vs. exploitation for various problem sizes.
    """

    def calculate_tour_distance(
        tour: List[int], distance_matrix: List[List[float]]
    ) -> float:
        """Calculates the total distance of a given tour."""
        total_distance = 0.0
        num_cities = len(tour)
        for i in range(num_cities):
            total_distance += distance_matrix[tour[i]][tour[(i + 1) % num_cities]]
        return total_distance

    def get_2opt_gain(
        tour: List[int], i: int, j: int, distance_matrix: List[List[float]]
    ) -> float:
        """
        Calculates the change in tour distance if a 2-opt swap is performed
        by reversing the segment between indices i and j (inclusive).
        """
        n = len(tour)
        if i == j:
            return 0.0
        if i > j:
            i, j = j, i

        prev_i_city = tour[(i - 1 + n) % n]
        curr_i_city = tour[i]
        curr_j_city = tour[j]
        next_j_city = tour[(j + 1) % n]

        old_edges_dist = (
            distance_matrix[prev_i_city][curr_i_city]
            + distance_matrix[curr_j_city][next_j_city]
        )
        new_edges_dist = (
            distance_matrix[prev_i_city][curr_j_city]
            + distance_matrix[curr_i_city][next_j_city]
        )
        return old_edges_dist - new_edges_dist

    def apply_2opt_swap(tour: List[int], i: int, j: int) -> List[int]:
        """Applies a 2-opt swap by reversing the segment between indices i and j (inclusive)."""
        if i > j:
            i, j = j, i
        new_tour = tour[:]
        new_tour[i : j + 1] = list(reversed(new_tour[i : j + 1]))
        return new_tour

    def two_opt_local_search(
        tour: List[int],
        distance_matrix: List[List[float]],
        max_iter_no_improve: int = 5,
    ) -> List[int]:
        """
        Performs 2-opt local search using delta evaluation, terminating after
        `max_iter_no_improve` passes without finding a better move.
        """
        n = len(tour)
        current_tour = list(tour)
        no_improve_count = 0

        while no_improve_count < max_iter_no_improve:
            best_gain_in_pass = 0.0
            best_i, best_j = -1, -1

            for i in range(n):
                # For 2-opt to be non-trivial, j must be at least i+2, and not i-1 (which would be no-op)
                # The segment is [i, i+1, ..., j]. Reversing requires at least 2 cities.
                # If i=0, j=n-1, it's (0,1,..,n-1) reversed -> (n-1,..,1,0) (full tour reversed)
                # A standard 2-opt cuts (tour[i-1], tour[i]) and (tour[j], tour[j+1]).
                # The segment to reverse is tour[i]...tour[j]. So j must be > i.
                # Minimal segment length 2 means j must be i+1. But this is trivial (swap two edges).
                # For non-trivial 2-opt (reversing at least one intermediate city), j >= i+2.
                for j in range(i + 2, n):
                    gain = get_2opt_gain(current_tour, i, j, distance_matrix)
                    if gain > best_gain_in_pass:
                        best_gain_in_pass = gain
                        best_i, best_j = i, j

            if (
                best_gain_in_pass > 1e-9
            ):  # Use a small epsilon for floating point comparison
                current_tour = apply_2opt_swap(current_tour, best_i, best_j)
                no_improve_count = 0
            else:
                no_improve_count += 1
        return current_tour

    def double_bridge_perturbation(tour: List[int]) -> List[int]:
        """
        Performs a double-bridge perturbation, a strong 4-opt equivalent move
        to escape local optima. Breaks 4 edges and reconnects them.
        """
        n = len(tour)
        if n < 8:  # Not enough cities for distinct segments, fallback to a 2-opt
            if n >= 2:
                new_tour = tour[:]
                i, j = random.sample(range(n), 2)
                if i > j:
                    i, j = j, i
                new_tour[i : j + 1] = list(reversed(new_tour[i : j + 1]))
                return new_tour
            return list(tour)

        # Randomly sample 4 distinct integers from 1 up to n-1.
        # These are cut points that divide the tour into 5 segments.
        idx = sorted(random.sample(range(1, n), 4))
        a, b, c, d = idx[0], idx[1], idx[2], idx[3]

        # Segments:
        segment_A = tour[0:a]
        segment_B = tour[a:b]
        segment_C = tour[b:c]
        segment_D = tour[c:d]
        segment_E = tour[d:n]

        # Reconstruct the tour using the double-bridge permutation: A + D + C + B + E
        new_tour = []
        new_tour.extend(segment_A)
        new_tour.extend(segment_D)
        new_tour.extend(segment_C)
        new_tour.extend(segment_B)
        new_tour.extend(segment_E)
        return new_tour

    def _apply_or_opt_move(
        tour: List[int],
        segment_start: int,
        segment_len: int,
        insert_pos: int,
        reversed_segment: bool,
    ) -> List[int]:
        """
        Applies an Or-opt move: removes a segment and re-inserts it.
        This function assumes insert_pos is the index of the city _before_ the insertion point.
        It also assumes that the segment is not inserted into its own indices (checked by gain function).
        """
        n = len(tour)

        # 1. Extract the segment cities and mark their original indices for removal.
        moved_segment_cities = []
        original_indices_of_segment = set()
        for k_offset in range(segment_len):
            idx = (segment_start + k_offset) % n
            moved_segment_cities.append(tour[idx])
            original_indices_of_segment.add(idx)

        if reversed_segment:
            moved_segment_cities.reverse()

        # 2. Build the tour without the moved segment, preserving relative order.
        tour_without_moved_segment = []
        for i in range(n):
            if i not in original_indices_of_segment:
                tour_without_moved_segment.append(tour[i])

        # 3. Determine the actual insertion point in `tour_without_moved_segment`.
        # The segment should be inserted after tour[insert_pos], i.e., before tour[(insert_pos + 1) % n].
        # We need to find the index of tour[(insert_pos + 1) % n] in `tour_without_moved_segment`.

        # If insert_pos is the last city (n-1), then (insert_pos + 1) % n is 0.
        # This means we insert before tour[0].

        # This `target_city_after_insertion` is guaranteed not to be in the moved segment
        # due to filtering in `get_or_opt_gain` and `random_or_opt_perturbation`.
        target_city_after_insertion = tour[(insert_pos + 1) % n]

        insertion_idx_in_temp = len(
            tour_without_moved_segment
        )  # Default to end of list
        try:
            insertion_idx_in_temp = tour_without_moved_segment.index(
                target_city_after_insertion
            )
        except ValueError:
            # Should not happen with robust checks; implies target city not found.
            # If `insert_pos` refers to the last city and segment_start=0,
            # `target_city_after_insertion` is `tour[0]`. If tour[0] is part of the segment,
            # this means `insert_pos` is (n-1) and segment includes 0. This is filtered.
            pass

        new_tour = (
            tour_without_moved_segment[:insertion_idx_in_temp]
            + moved_segment_cities
            + tour_without_moved_segment[insertion_idx_in_temp:]
        )

        return new_tour

    def get_or_opt_gain(
        tour: List[int],
        segment_start: int,
        segment_len: int,
        insert_pos: int,
        reversed_segment: bool,
        distance_matrix: List[List[float]],
        n: int,
    ) -> float:
        """
        Calculates the change in tour distance for a single Or-opt move (segment relocation).
        `insert_pos` is the index of the city *before* the insertion point.
        """

        # Original edges (3 broken):
        prev_s_city = tour[(segment_start - 1 + n) % n]
        curr_s_city = tour[segment_start]  # First city of segment

        end_s_idx = (segment_start + segment_len - 1) % n
        curr_e_city = tour[end_s_idx]  # Last city of segment
        next_s_city = tour[(end_s_idx + 1) % n]  # City after segment

        p_city = tour[insert_pos]  # City before insertion point
        next_p_city = tour[(insert_pos + 1) % n]  # City after insertion point

        old_dist = (
            distance_matrix[prev_s_city][curr_s_city]
            + distance_matrix[curr_e_city][next_s_city]
            + distance_matrix[p_city][next_p_city]
        )

        # Cities that will be at the start/end of the *moved* segment
        moved_segment_first_city = curr_s_city if not reversed_segment else curr_e_city
        moved_segment_last_city = curr_e_city if not reversed_segment else curr_s_city

        # Calculate new total edge distance
        new_dist = (
            distance_matrix[prev_s_city][next_s_city]
            + distance_matrix[p_city][moved_segment_first_city]
            + distance_matrix[moved_segment_last_city][next_p_city]
        )

        return old_dist - new_dist

    def random_or_opt_perturbation(tour: List[int], max_segment_len: int) -> List[int]:
        """
        Applies a single *random* Or-opt move (segment relocation) for perturbation.
        Randomly picks a segment and a new insertion point, and decides on reversal.
        """
        n = len(tour)
        # Or-opt requires at least 4 cities for non-trivial moves (e.g., segment of length 1, 2, or 3)
        if n < 4:
            return list(tour)

        attempts = 0
        max_attempts = n * 2  # Limit attempts to find a valid, non-trivial move

        while attempts < max_attempts:
            segment_len = random.randint(1, min(max_segment_len, n - 1))
            segment_start = random.randrange(n)
            # This is the city *before* insertion point
            insert_pos = random.randrange(n)

            segment_indices = set((segment_start + k) % n for k in range(segment_len))

            # Check for trivial moves or insertion into self/adjacent to original:
            # If insertion point is within the segment, or immediately adjacent
            if (
                insert_pos in segment_indices
                or (insert_pos + 1) % n in segment_indices
                or insert_pos == (segment_start - 1 + n) % n
                or (insert_pos + 1) % n == (segment_start + segment_len) % n
            ):
                attempts += 1
                continue

            break  # Valid non-trivial move found

        if attempts >= max_attempts:
            # If no valid move found after attempts, return original tour
            return list(tour)

        # Randomly decide to reverse segment or not
        reversed_segment = random.choice([True, False])

        return _apply_or_opt_move(
            tour, segment_start, segment_len, insert_pos, reversed_segment
        )

    def randomized_or_opt_local_search_pass(
        tour: List[int],
        distance_matrix: List[List[float]],
        max_segment_len: int,
        num_segment_samples: int,
        num_insert_samples: int,
    ) -> List[int]:
        """
        Performs a single pass of a randomized Or-opt local search.
        It samples a limited number of segments and insertion points
        and applies the best improving move found among these samples.
        This is an O(K) operation where K is a fixed sample size.
        """
        n = len(tour)
        if n < 4:
            return list(tour)

        current_tour = list(tour)
        best_gain = 0.0
        best_move = None

        # Sample starting points for segments
        segments_to_try_indices = random.sample(range(n), min(n, num_segment_samples))

        for segment_start in segments_to_try_indices:
            for segment_len in range(1, min(max_segment_len + 1, n)):
                if segment_len >= n:
                    continue  # Cannot move segment as long as or longer than tour

                # Sample insertion points
                insert_points_to_try_indices = random.sample(
                    range(n), min(n, num_insert_samples)
                )

                for insert_pos in insert_points_to_try_indices:
                    # Check for trivial moves:
                    segment_indices = set(
                        (segment_start + k) % n for k in range(segment_len)
                    )
                    if (
                        insert_pos in segment_indices
                        or (insert_pos + 1) % n in segment_indices
                        or insert_pos == (segment_start - 1 + n) % n
                        or (insert_pos + 1) % n == (segment_start + segment_len) % n
                    ):
                        continue

                    for reversed_segment in [False, True]:
                        gain = get_or_opt_gain(
                            current_tour,
                            segment_start,
                            segment_len,
                            insert_pos,
                            reversed_segment,
                            distance_matrix,
                            n,
                        )

                        if gain > best_gain:
                            best_gain = gain
                            best_move = (
                                segment_start,
                                segment_len,
                                insert_pos,
                                reversed_segment,
                            )

        if best_gain > 1e-9:
            segment_start, segment_len, insert_pos, reversed_segment = best_move
            return _apply_or_opt_move(
                current_tour, segment_start, segment_len, insert_pos, reversed_segment
            )

        return current_tour

    if number_of_city < 2:
        return child[:]

    mutated_tour = list(child)

    # Phase 1: Initial Deep Local Search (Exploitation) - Adaptive parameters
    initial_two_opt_depth = 0
    if number_of_city <= 50:
        initial_two_opt_depth = 25  # Very deep search for small instances
    elif number_of_city <= 150:
        initial_two_opt_depth = 15  # Robust depth for medium instances
    else:
        initial_two_opt_depth = (
            8  # Shallower but still effective for very large instances
        )

    mutated_tour = two_opt_local_search(
        mutated_tour, distance_matrix, max_iter_no_improve=initial_two_opt_depth
    )
    current_best_distance = calculate_tour_distance(mutated_tour, distance_matrix)

    # Phase 2: Iterated Local Search (ILS) - Perturbation and Re-optimization
    # Tuned parameters for MAX_ILS_ITERATIONS and PATIENCE
    MAX_ILS_ITERATIONS = 0
    PATIENCE = 0

    if number_of_city <= 50:
        MAX_ILS_ITERATIONS = 150
        PATIENCE = 30
    elif number_of_city <= 150:
        MAX_ILS_ITERATIONS = 200
        PATIENCE = 40
    elif number_of_city <= 500:
        MAX_ILS_ITERATIONS = 300
        PATIENCE = 60
    else:  # Very large instances
        MAX_ILS_ITERATIONS = 400
        PATIENCE = 80

    global_best_tour = list(mutated_tour)
    global_best_distance = current_best_distance

    patience_counter = PATIENCE

    for _ in range(MAX_ILS_ITERATIONS):
        perturbed_tour = list(mutated_tour)

        # Adaptive Perturbation Strategy:
        # Use random Or-opt for smaller N (stronger, more structural disruption than multi-2-opt)
        # Use Double Bridge for larger N (classic strong structural perturbation)
        perturbation_max_segment_len = 3  # Default for Or-opt perturbation

        if number_of_city <= 75:
            # For very small N, a more aggressive Or-opt perturbation
            perturbed_tour = random_or_opt_perturbation(
                perturbed_tour, max_segment_len=min(number_of_city - 1, 4)
            )
        elif number_of_city <= 200:
            # Standard Or-opt perturbation for medium N
            perturbed_tour = random_or_opt_perturbation(
                perturbed_tour, max_segment_len=3
            )
        else:
            # Double Bridge for larger N, as it's a fixed-cost strong perturbation
            perturbed_tour = double_bridge_perturbation(perturbed_tour)

        # Re-optimization: Apply a shallower 2-opt local search
        re_opt_max_no_improve = 0
        if number_of_city <= 100:
            re_opt_max_no_improve = 4  # Slightly deeper 2-opt for smaller instances
        elif number_of_city <= 300:
            re_opt_max_no_improve = 3  # Standard 2-opt depth
        else:
            re_opt_max_no_improve = 2  # Shallower 2-opt for very large instances

        re_optimized_tour = two_opt_local_search(
            perturbed_tour, distance_matrix, max_iter_no_improve=re_opt_max_no_improve
        )

        # Adaptive Randomized Or-opt Pass for further fast local improvement
        or_opt_pass_max_segment_len = 0
        or_opt_pass_num_segment_samples = 0
        or_opt_pass_num_insert_samples = 0

        if number_of_city <= 75:
            or_opt_pass_max_segment_len = 3
            or_opt_pass_num_segment_samples = min(number_of_city, 20)
            or_opt_pass_num_insert_samples = min(number_of_city, 20)
        elif number_of_city <= 200:
            or_opt_pass_max_segment_len = 3
            or_opt_pass_num_segment_samples = min(number_of_city, 15)
            or_opt_pass_num_insert_samples = min(number_of_city, 15)
        elif number_of_city <= 500:
            or_opt_pass_max_segment_len = 2  # Limit segment length for larger N
            or_opt_pass_num_segment_samples = min(number_of_city, 10)
            or_opt_pass_num_insert_samples = min(number_of_city, 10)
        else:  # For very large N, keep Or-opt pass very fast
            or_opt_pass_max_segment_len = 1  # Only single city moves
            or_opt_pass_num_segment_samples = min(
                number_of_city, 5
            )  # Small constant samples
            or_opt_pass_num_insert_samples = min(number_of_city, 5)

        # Apply the randomized Or-opt pass
        re_optimized_tour = randomized_or_opt_local_search_pass(
            re_optimized_tour,
            distance_matrix,
            max_segment_len=or_opt_pass_max_segment_len,
            num_segment_samples=or_opt_pass_num_segment_samples,
            num_insert_samples=or_opt_pass_num_insert_samples,
        )

        re_optimized_distance = calculate_tour_distance(
            re_optimized_tour, distance_matrix
        )

        # Acceptance criteria (greedy for current_best, and global_best)
        if re_optimized_distance < global_best_distance:
            global_best_tour = re_optimized_tour
            global_best_distance = re_optimized_distance
            mutated_tour = (
                re_optimized_tour  # Set new base for ILS if global improvement
            )
            patience_counter = PATIENCE  # Reset patience on global improvement
        elif re_optimized_distance < current_best_distance:
            mutated_tour = re_optimized_tour
            current_best_distance = re_optimized_distance
            patience_counter = PATIENCE  # Reset patience
        else:
            patience_counter -= 1
            if patience_counter <= 0:
                break  # Terminate ILS early if patience runs out

    return global_best_tour
