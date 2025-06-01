import numpy as np
import random
import math
import scipy
import torch
def mutate_v2(number_of_city: int, child: List[int], distance_matrix: List[List[float]],
              # --- Local Search Operator Parameters ---
              or_opt_max_segment_size: int = 4.192526548126102,

              # --- Perturbation Operator Parameters ---
              reinsert_max_block_size: int = 2.6857279739454984,
              scramble_min_cities: int = 3.6044347437588513,
              double_bridge_min_cities: int = 9.903351923849039,

              # --- VND Sequence Parameters ---
              or_opt_min_cities_for_vnd: int = 7.053251884224734,
              three_opt_max_cities_for_vnd: int = 97.17380544096517,

              # --- Adaptive ILS Control Parameters (based on problem size tiers) ---
              ils_threshold_1_cities: int = 30.768649966968404,
              ils_iterations_1: int = 6.526634118271622,
              ils_stagnation_1: int = 4.791829479962004,

              ils_threshold_2_cities: int = 88.25707627928016,
              ils_iterations_2: int = 10.494012680417374,
              ils_stagnation_2: int = 3.2396483114396437,

              ils_threshold_3_cities: int = 329.8857686123612,
              ils_iterations_3: int = 15.262377468386369,
              ils_stagnation_3: int = 6.792603130873717,
              
              ils_iterations_large_problems: int = 21.44163875327562,
              ils_stagnation_large_problems: int = 10.440251943819076,
              ) -> List[int]:
    """
    Mutates a TSP tour using an advanced Adaptive Iterated Local Search (ILS) framework
    with Variable Neighborhood Descent (VND) for local search.
    It adaptively selects local search operators (2-opt, 3-opt, Or-opt) and perturbation operators
    (segment reinsertion, scramble, double bridge) based on problem size and search progress
    to balance exploration and exploitation, aiming to escape local optima more effectively.
    
    Parameters
    ----------
    number_of_city : int
        The total number of cities in the TSP problem.
    child : List[int]
        The current tour (solution) to be mutated.
    distance_matrix : List[List[float]]
        The matrix containing distances between cities.
    or_opt_max_segment_size : int, optional
        Maximum segment size (k) for Or-opt (k-opt) local search. A segment of up to this size
        can be moved to another position in the tour. By default 3.
    reinsert_max_block_size : int, optional
        Maximum block size for segment reinsertion perturbation. A random segment of up to
        this size is removed and reinserted. By default 3.
    scramble_min_cities : int, optional
        Minimum number of cities required for the scramble perturbation to be applied.
        If the tour is smaller than this, scramble perturbation is skipped. By default 3.
    double_bridge_min_cities : int, optional
        Minimum number of cities required for the double bridge perturbation to be applied.
        If the tour is smaller than this, double bridge perturbation is skipped. By default 8.
    or_opt_min_cities_for_vnd : int, optional
        Minimum number of cities for Or-opt to be included in the Variable Neighborhood Descent (VND)
        sequence of local search operators. Or-opt is computationally more expensive. By default 10.
    three_opt_max_cities_for_vnd : int, optional
        Maximum number of cities for 3-opt to be included in the VND sequence. 3-opt is
        computationally very intensive and might be too slow for very large problems. By default 150.
    ils_threshold_1_cities : int, optional
        City count threshold for the first tier of Adaptive ILS parameters. Applies to problems
        with `number_of_city <= ils_threshold_1_cities`. By default 30.
    ils_iterations_1 : int, optional
        Number of ILS iterations for problems in the first tier. By default 7.
    ils_stagnation_1 : int, optional
        Maximum consecutive non-improving steps before a strong perturbation is applied
        for problems in the first tier. By default 2.
    ils_threshold_2_cities : int, optional
        City count threshold for the second tier of Adaptive ILS parameters. Applies to problems
        with `ils_threshold_1_cities < number_of_city <= ils_threshold_2_cities`. By default 100.
    ils_iterations_2 : int, optional
        Number of ILS iterations for problems in the second tier. By default 10.
    ils_stagnation_2 : int, optional
        Maximum consecutive non-improving steps before a strong perturbation is applied
        for problems in the second tier. By default 4.
    ils_threshold_3_cities : int, optional
        City count threshold for the third tier of Adaptive ILS parameters. Applies to problems
        with `ils_threshold_2_cities < number_of_city <= ils_threshold_3_cities`. By default 300.
    ils_iterations_3 : int, optional
        Number of ILS iterations for problems in the third tier. By default 12.
    ils_stagnation_3 : int, optional
        Maximum consecutive non-improving steps before a strong perturbation is applied
        for problems in the third tier. By default 6.
    ils_iterations_large_problems : int, optional
        Number of ILS iterations for problems larger than `ils_threshold_3_cities`. By default 15.
    ils_stagnation_large_problems : int, optional
        Maximum consecutive non-improving steps before a strong perturbation is applied
        for problems larger than `ils_threshold_3_cities`. By default 8.

    Returns
    -------
    List[int]
        The best mutated TSP tour found.
    """

    if number_of_city < 2:
        return child[:] # Cannot mutate with less than 2 cities
