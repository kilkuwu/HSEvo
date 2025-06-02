import random
from typing import List


def mutate_v2(
    number_of_city: int, child: List[int], distance_matrix: List[List[float]]
) -> List[int]:
    def calculate_tour_distance(
        tour: List[int], distance_matrix: List[List[float]]
    ) -> float:
        total_distance = 0.0
        num_cities = len(tour)
        for i in range(num_cities):
            total_distance += distance_matrix[tour[i]][tour[(i + 1) % num_cities]]
        return total_distance

    def get_2opt_gain(
        tour: List[int], i: int, j: int, distance_matrix: List[List[float]]
    ) -> float:
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
        n = len(tour)
        current_tour = list(tour)
        no_improve_count = 0

        while no_improve_count < max_iter_no_improve:
            best_gain_in_pass = 0.0
            best_i, best_j = -1, -1

            for i in range(n):
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

        idx = sorted(random.sample(range(1, n), 4))
        a, b, c, d = idx[0], idx[1], idx[2], idx[3]

        segment_A = tour[0:a]
        segment_B = tour[a:b]
        segment_C = tour[b:c]
        segment_D = tour[c:d]
        segment_E = tour[d:n]

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
        n = len(tour)

        moved_segment_cities = []
        original_indices_of_segment = set()
        for k_offset in range(segment_len):
            idx = (segment_start + k_offset) % n
            moved_segment_cities.append(tour[idx])
            original_indices_of_segment.add(idx)

        if reversed_segment:
            moved_segment_cities.reverse()

        tour_without_moved_segment = []
        for i in range(n):
            if i not in original_indices_of_segment:
                tour_without_moved_segment.append(tour[i])

        target_city_after_insertion = tour[(insert_pos + 1) % n]

        insertion_idx_in_temp = len(
            tour_without_moved_segment
        )  # Default to end of list
        try:
            insertion_idx_in_temp = tour_without_moved_segment.index(
                target_city_after_insertion
            )
        except ValueError:
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

        moved_segment_first_city = curr_s_city if not reversed_segment else curr_e_city
        moved_segment_last_city = curr_e_city if not reversed_segment else curr_s_city

        new_dist = (
            distance_matrix[prev_s_city][next_s_city]
            + distance_matrix[p_city][moved_segment_first_city]
            + distance_matrix[moved_segment_last_city][next_p_city]
        )

        return old_dist - new_dist

    def random_or_opt_perturbation(tour: List[int], max_segment_len: int) -> List[int]:
        n = len(tour)
        if n < 4:
            return list(tour)

        attempts = 0
        max_attempts = n * 2  # Limit attempts to find a valid, non-trivial move

        while attempts < max_attempts:
            segment_len = random.randint(1, min(max_segment_len, n - 1))
            segment_start = random.randrange(n)
            insert_pos = random.randrange(n)

            segment_indices = set((segment_start + k) % n for k in range(segment_len))

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
            return list(tour)

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
        n = len(tour)
        if n < 4:
            return list(tour)

        current_tour = list(tour)
        best_gain = 0.0
        best_move = None

        segments_to_try_indices = random.sample(range(n), min(n, num_segment_samples))

        for segment_start in segments_to_try_indices:
            for segment_len in range(1, min(max_segment_len + 1, n)):
                if segment_len >= n:
                    continue  # Cannot move segment as long as or longer than tour

                insert_points_to_try_indices = random.sample(
                    range(n), min(n, num_insert_samples)
                )

                for insert_pos in insert_points_to_try_indices:
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

        perturbation_max_segment_len = 3  # Default for Or-opt perturbation

        if number_of_city <= 75:
            perturbed_tour = random_or_opt_perturbation(
                perturbed_tour, max_segment_len=min(number_of_city - 1, 4)
            )
        elif number_of_city <= 200:
            perturbed_tour = random_or_opt_perturbation(
                perturbed_tour, max_segment_len=3
            )
        else:
            perturbed_tour = double_bridge_perturbation(perturbed_tour)

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
