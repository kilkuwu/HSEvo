import numpy as np
import random
import math
import scipy
import torch
def mutate_v2(
    number_of_city: int,
    child: List[int],
    distance_matrix: List[List[float]],
    or_opt_max_segment_length: int = 3.450310279358075,
    min_cities_for_double_bridge: int = 7.599443576783526,
    max_vns_outer_iterations: int = 17.25238410791451) -> List[int]:
    """
    Mutates a tour using a Variable Neighborhood Search (VNS) framework.
    Applies initial local search, then iteratively perturbs the solution
    with increasing intensity (OR-opt, Double Bridge) and re-optimizes
    using 2-opt hill climbing.

    Args:
        number_of_city (int): The total number of cities in the TSP.
        child (List[int]): The current tour (solution) to be mutated.
        distance_matrix (List[List[float]]): The matrix storing distances between cities.
        or_opt_max_segment_length (int): Maximum length of the segment to move in OR-opt.
                                         The actual length is chosen randomly from 1 to this value.
        min_cities_for_double_bridge (int): Minimum number of cities required to perform a
                                            double bridge perturbation.
        max_vns_outer_iterations (int): The maximum number of times to cycle through the
                                        neighborhoods (perturbation operators) in the VNS loop.

    Returns:
        List[int]: An improved tour after applying VNS-based mutation.
    """

    def objective_function(tour: List[int]) -> float:
        """Calculate the total distance of a tour."""
        if not tour:
            return float('inf')
