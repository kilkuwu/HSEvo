#!/usr/bin/env python3
"""
High-Performance Branch and Bound TSP Solver
Implements multiple optimization techniques for solving TSP instances.
"""

import sys
import heapq
import time
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass


@dataclass
class Node:
    """Represents a node in the branch and bound tree."""
    cost: float
    level: int
    path: List[int]
    visited: Set[int]
    reduced_matrix: List[List[float]]
    
    def __lt__(self, other):
        # Priority queue ordering: lower cost first, then higher level (deeper search)
        if abs(self.cost - other.cost) < 1e-9:
            return self.level > other.level
        return self.cost < other.cost


class TSPBranchAndBound:
    """Advanced Branch and Bound TSP Solver with multiple optimizations."""
    
    def __init__(self, distance_matrix: List[List[float]]):
        self.n = len(distance_matrix)
        self.distance_matrix = [row[:] for row in distance_matrix]
        self.best_cost = float('inf')
        self.best_path = []        self.nodes_explored = 0
        self.nodes_pruned = 0
        
        # Replace large values (diagonal) with infinity for proper handling
        for i in range(self.n):
            for j in range(self.n):
                if self.distance_matrix[i][j] > 1e8:
                    self.distance_matrix[i][j] = float('inf')
    
    def reduce_matrix(self, matrix: List[List[float]]) -> Tuple[List[List[float]], float]:
        """
        Reduce the cost matrix by subtracting minimum values from rows and columns.
        Returns the reduced matrix and the reduction cost.
        """
        n = len(matrix)
        reduced = [row[:] for row in matrix]
        reduction_cost = 0.0
        
        # Row reduction
        for i in range(n):
            finite_values = [val for val in reduced[i] if val != float('inf')]
            if finite_values:
                row_min = min(finite_values)
                if row_min > 0:
                    reduction_cost += row_min
                    for j in range(n):
                        if reduced[i][j] != float('inf'):
                            reduced[i][j] -= row_min
        
        # Column reduction
        for j in range(n):
            finite_values = [reduced[i][j] for i in range(n) if reduced[i][j] != float('inf')]
            if finite_values:
                col_min = min(finite_values)                if col_min > 0:
                    reduction_cost += col_min
                    for i in range(n):
                        if reduced[i][j] != float('inf'):
                            reduced[i][j] -= col_min
        
        return reduced, reduction_cost
    
    def calculate_bound(self, node: Node, next_city: int) -> Tuple[List[List[float]], float]:
        """
        Calculate the lower bound for a node when moving to next_city.
        Returns the new reduced matrix and the bound cost.
        """
        if len(node.path) == 0:
            current_city = 0  # Start from city 0
        else:
            current_city = node.path[-1]
        
        # Create new matrix by setting appropriate cells to infinity
        new_matrix = [row[:] for row in node.reduced_matrix]
        
        # Set row of current city to infinity
        for j in range(self.n):
            new_matrix[current_city][j] = float('inf')
        
        # Set column of next city to infinity
        for i in range(self.n):
            new_matrix[i][next_city] = float('inf')
        
        # Prevent subtours by setting return edge to infinity (except for last step)
        if len(node.path) > 0 and len(node.path) < self.n - 1:
            new_matrix[next_city][node.path[0]] = float('inf')
        
        # Reduce the matrix and get the reduction cost
        reduced_matrix, reduction_cost = self.reduce_matrix(new_matrix)
        
        # Calculate the total bound
        edge_cost = node.reduced_matrix[current_city][next_city]
        if edge_cost == float('inf'):
            return reduced_matrix, float('inf')
        
        total_cost = node.cost + edge_cost + reduction_cost
        
        return reduced_matrix, total_cost
    
    def get_nearest_neighbor_bound(self, start_city: int = 0) -> float:
        """Get a quick upper bound using nearest neighbor heuristic."""
        visited = set([start_city])
        current = start_city
        total_cost = 0.0
        
        for _ in range(self.n - 1):
            min_cost = float('inf')
            next_city = -1
            
            for j in range(self.n):
                if j not in visited and self.distance_matrix[current][j] < min_cost:
                    min_cost = self.distance_matrix[current][j]
                    next_city = j
            
            if next_city != -1:
                total_cost += min_cost
                visited.add(next_city)
                current = next_city
        
        # Return to start
        total_cost += self.distance_matrix[current][start_city]
        return total_cost
    
    def get_mst_bound(self, unvisited: Set[int], last_city: int) -> float:
        """
        Calculate MST-based lower bound for remaining unvisited cities.
        """
        if len(unvisited) <= 1:
            return 0.0
        
        # Find minimum spanning tree for unvisited cities
        cities = list(unvisited)
        if len(cities) == 0:
            return 0.0
        
        # Prim's algorithm for MST
        mst_cost = 0.0
        in_mst = set([cities[0]])
        
        while len(in_mst) < len(cities):
            min_edge = float('inf')
            for u in in_mst:
                for v in cities:
                    if v not in in_mst and self.distance_matrix[u][v] < min_edge:
                        min_edge = self.distance_matrix[u][v]
                        next_vertex = v
              if min_edge != float('inf'):
                mst_cost += min_edge
                in_mst.add(next_vertex)
            else:
                break
        
        # Add minimum cost to connect last visited city to unvisited set
        costs_to_unvisited = [self.distance_matrix[last_city][city] for city in unvisited 
                             if self.distance_matrix[last_city][city] != float('inf')]
        if costs_to_unvisited:
            min_to_unvisited = min(costs_to_unvisited)
            mst_cost += min_to_unvisited
        
        return mst_cost
    
    def solve(self, time_limit: float = 300.0) -> Tuple[float, List[int]]:
        """
        Solve TSP using branch and bound with multiple optimizations.
        
        Args:
            time_limit: Maximum time in seconds (default 5 minutes)
            
        Returns:
            Tuple of (best_cost, best_path)
        """
        start_time = time.time()
        
        # Get initial upper bound using nearest neighbor
        self.best_cost = self.get_nearest_neighbor_bound()
        
        # Try multiple starting points for better initial bound
        for start in range(min(self.n, 5)):
            nn_cost = self.get_nearest_neighbor_bound(start)
            if nn_cost < self.best_cost:
                self.best_cost = nn_cost
        
        print(f"Initial upper bound (Nearest Neighbor): {self.best_cost:.2f}")
        
        # Initialize the root node
        reduced_matrix, initial_cost = self.reduce_matrix(self.distance_matrix)
        root = Node(
            cost=initial_cost,
            level=0,
            path=[],
            visited=set(),
            reduced_matrix=reduced_matrix
        )
          # Priority queue for branch and bound
        pq = [root]
        heapq.heapify(pq)
        
        while pq and time.time() - start_time < time_limit:
            current = heapq.heappop(pq)
            self.nodes_explored += 1
            
            # Progress reporting
            if self.nodes_explored % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Explored {self.nodes_explored} nodes, best: {self.best_cost:.2f}, "
                      f"time: {elapsed:.1f}s, queue: {len(pq)}")
            
            # Pruning: if current node's bound is >= best known solution (with small tolerance)
            if current.cost > self.best_cost + 1e-9:
                self.nodes_pruned += 1
                continue
            
            # If we have visited all cities
            if len(current.path) == self.n:
                # Complete the tour by returning to start
                if len(current.path) > 0:
                    return_cost = self.distance_matrix[current.path[-1]][current.path[0]]
                    total_cost = current.cost + return_cost
                    
                    if total_cost < self.best_cost:
                        self.best_cost = total_cost
                        self.best_path = current.path[:]
                        print(f"New best solution: {self.best_cost:.2f}")
                continue
            
            # Determine current city
            if len(current.path) == 0:
                current_city = 0  # Start from city 0
                next_path = [0]
                next_visited = set([0])
            else:
                current_city = current.path[-1]
                next_path = current.path[:]
                next_visited = current.visited.copy()
            
            # Branch: try all unvisited cities
            candidates = []
            for next_city in range(self.n):
                if next_city not in current.visited and next_city != current_city:
                    new_matrix, bound_cost = self.calculate_bound(current, next_city)
                    
                    # Additional MST-based bound for tighter estimates
                    remaining = set(range(self.n)) - next_visited - {next_city}
                    if len(remaining) > 0:
                        mst_bound = self.get_mst_bound(remaining, next_city)
                        bound_cost += mst_bound
                    
                    if bound_cost < self.best_cost:
                        candidates.append((bound_cost, next_city, new_matrix))
            
            # Sort candidates by bound (best first) for better pruning
            candidates.sort()
            
            # Add promising candidates to priority queue
            for bound_cost, next_city, new_matrix in candidates:
                if bound_cost < self.best_cost:
                    new_path = next_path + [next_city]
                    new_visited = next_visited | {next_city}
                    
                    child = Node(
                        cost=bound_cost,
                        level=current.level + 1,
                        path=new_path,
                        visited=new_visited,
                        reduced_matrix=new_matrix
                    )
                    heapq.heappush(pq, child)
        
        elapsed_time = time.time() - start_time
        print(f"\nSearch completed in {elapsed_time:.2f} seconds")
        print(f"Nodes explored: {self.nodes_explored}")
        print(f"Nodes pruned: {self.nodes_pruned}")
        print(f"Final queue size: {len(pq)}")
        
        return self.best_cost, self.best_path


def load_tsp_instance(filename: str) -> List[List[float]]:
    """Load TSP instance from file in the tsp_dataset format."""
    with open(filename, 'r') as f:
        lines = f.readlines()
        n_cities = int(lines[0].strip())
        
        distance_matrix = []
        for i in range(1, n_cities + 1):
            row = list(map(float, lines[i].strip().split()))
            distance_matrix.append(row)
    
    return distance_matrix


def main():
    """Main function for standalone execution."""
    if len(sys.argv) != 2:
        print("Usage: python tsp_branch_and_bound.py <instance_file>")
        print("Example: python tsp_branch_and_bound.py tsp_dataset/val/10_00.inp")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        # Load the TSP instance
        print(f"Loading TSP instance from: {filename}")
        distance_matrix = load_tsp_instance(filename)
        n_cities = len(distance_matrix)
        print(f"Problem size: {n_cities} cities")
        
        # Create solver and solve
        solver = TSPBranchAndBound(distance_matrix)
        
        # Set time limit based on problem size
        if n_cities <= 10:
            time_limit = 60.0  # 1 minute for small instances
        elif n_cities <= 15:
            time_limit = 300.0  # 5 minutes for medium instances
        else:
            time_limit = 600.0  # 10 minutes for large instances
        
        print(f"Time limit: {time_limit} seconds")
        print("Starting branch and bound search...")
        
        best_cost, best_path = solver.solve(time_limit)
        
        print(f"\n=== SOLUTION ===")
        print(f"Best tour cost: {best_cost:.2f}")
        print(f"Best path: {' -> '.join(map(str, best_path))} -> {best_path[0] if best_path else 0}")
        
        # Verify the solution
        if best_path:
            verified_cost = 0.0
            for i in range(len(best_path)):
                from_city = best_path[i]
                to_city = best_path[(i + 1) % len(best_path)]
                verified_cost += distance_matrix[from_city][to_city]
            print(f"Verified cost: {verified_cost:.2f}")
            
            if abs(verified_cost - best_cost) > 1e-6:
                print("WARNING: Cost verification failed!")
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()