from typing import List

def heuristics_v2(distance_matrix: List[List[float]]) -> List[List[float]]:
    """
    Compute heuristic values for ACO, incorporating distance and a biased "attractiveness" factor.

    Args:
        distance_matrix: 2D list of distances between cities

    Returns:
        heuristic_matrix: 2D list of heuristic values
    """
    n = len(distance_matrix)
    heuristic = [[0.0] * n for _ in range(n)]

    # Calculate average distance for each city
    avg_distances = [sum(distance_matrix[i]) / (n - 1) if n > 1 else 0 for i in range(n)]


    for i in range(n):
        for j in range(n):
            if i != j and distance_matrix[i][j] > 0:
                # Inverse distance, but also consider proximity to other cities.

                attractiveness_factor = avg_distances[j] / (distance_matrix[i][j] + 1e-9) # Smaller jth city average distance to all other cities --> increase heuritic val

                heuristic[i][j] = (1.0 / distance_matrix[i][j]) * attractiveness_factor
            else:
                heuristic[i][j] = 0.0

    return heuristic
