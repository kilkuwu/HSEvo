from typing import List

def heuristics_v2(distance_matrix: List[List[float]]) -> List[List[float]]:
    """
    Compute heuristic values (inverse of distance) for ACO algorithm.
    Args:
        distance_matrix: 2D list of distances between cities
    Returns:
        heuristic_matrix: 2D list of heuristic values (1/distance)
    """
    n = len(distance_matrix)
    heuristic = []
    
    for i in range(n):
        row = []
        for j in range(n):
            if distance_matrix[i][j] > 0:
                row.append(1.0 / distance_matrix[i][j])
            else:
                row.append(0.0)
        heuristic.append(row)
    
    return heuristic
