from os import path
import sys
import random
import inspect
import gpt
import time
from aco import ACO

def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name

possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)

def solve(dist_mat):
    """Solve TSP using ACO with evolved heuristics function."""
    # Set diagonal to a large number
    for i in range(len(dist_mat)):
        dist_mat[i][i] = int(1e9)
    
    # Create heuristic matrix using the evolved function
    heu = heuristics(dist_mat)
    
    # Add small epsilon to avoid zero values
    for i in range(len(heu)):
        for j in range(len(heu[i])):
            if heu[i][j] < 1e-9:
                heu[i][j] = 1e-9
            else:
                heu[i][j] += 1e-9
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Configure ACO parameters based on problem size
    n_cities = len(dist_mat)
    if n_cities <= 20:
        n_ants = 100
        n_iterations = 500
    elif n_cities <= 50:
        n_ants = 100
        n_iterations = 200
    elif n_cities <= 100:
        n_ants = 50
        n_iterations = 100
    else:
        n_ants = 50
        n_iterations = 100
    
    aco = ACO(dist_mat, heu, n_ants=n_ants)
    best_distance = aco.run(n_iterations)
    return best_distance

def calculate_mean(values):
    """Calculate mean without numpy."""
    if not values:
        return 0.0
    return sum(values) / len(values)

if __name__ == "__main__":
    print("[*] Running TSP ACO evaluation...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']

    # Use the existing TSP dataset
    dataset_dir = path.join(path.dirname(__file__), "../../tsp_dataset")
    
    if mood == 'train':
        # Load training instances
        objs = []
        num_instances = 16  # Use 16 instances for training
        dataset_dir = path.join(dataset_dir, "train")
        
        for i in range(num_instances):
            instance_file = path.join(dataset_dir, f"{problem_size}_{i:02d}.inp")
            if path.isfile(instance_file):
                # Load distance matrix from file
                with open(instance_file, 'r') as f:
                    lines = f.readlines()
                    n_cities = int(lines[0].strip())
                    dist_mat = []
                    for j in range(1, n_cities + 1):
                        row = list(map(float, lines[j].strip().split()))
                        dist_mat.append(row)
                
                # Solve instance
                obj = solve(dist_mat)
                print(f"[*] Instance {i}: {obj}")
                objs.append(obj)
            else:
                print(f"[*] Warning: Instance file {instance_file} not found")
        
        if objs:
            print("[*] Average:")
            print(calculate_mean(objs))
        else:
            print("[*] No valid instances found")
            print(float('inf'))  # Return worst possible score
    
    else:
        dataset_dir = path.join(dataset_dir, "val")
        results = []
        for test_size in [10, 20, 50, 100, 200]:
            times = []
            objs = []
            num_instances = 16  # Use fewer instances for validation
            
            for i in range(num_instances):
                instance_file = path.join(dataset_dir, f"{test_size}_{i:02d}.inp")
                if path.isfile(instance_file):
                    # Load distance matrix from file
                    with open(instance_file, 'r') as f:
                        lines = f.readlines()
                        n_cities = int(lines[0].strip())
                        dist_mat = []
                        for j in range(1, n_cities + 1):
                            row = list(map(float, lines[j].strip().split()))
                            dist_mat.append(row)
                    
                    # Solve instance
                    start_time = time.time()
                    obj = solve(dist_mat)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    times.append(elapsed_time)
                    print(
                        f"> Instance {i} took {elapsed_time:.4f}s, result: {obj}")
                    objs.append(obj)
                else:
                    print(f"[*] Warning: Instance file {instance_file} not found")
            
            if objs:
                result = calculate_mean(objs)
                time_mean = calculate_mean(times)
                print(f"[*] Average obj for {test_size}: {result}")
                print(
                    f"[*] Total: {sum(times):.4f}s, Average: {time_mean:.4f}s")
                results.append((test_size, result, time_mean))
        print("[*] Final Results:")
        for size, avg, tim in results:
            print(f"Size {size}: {avg} in {tim:.4f}s")
        