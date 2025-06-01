from os import path
from aco import ACO
import sys
import gpt
import inspect

def get_heuristic_name(module, possible_names):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name

def calculate_mean(values):
    """Calculate mean without numpy."""
    if not values:
        return 0.0
    return sum(values) / len(values)

possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)

N_ITERATIONS = 200  # Reduced for PyPy compatibility and speed
N_ANTS = 50  # Reduced for better performance


def solve(dist_mat):
    # Set diagonal to a large number
    for i in range(len(dist_mat)):
        dist_mat[i][i] = int(1e9)
    
    # Create heuristic matrix
    heu = heuristics(dist_mat)
    
    # Add small epsilon to avoid zero values
    for i in range(len(heu)):
        for j in range(len(heu[i])):
            if heu[i][j] < 1e-9:
                heu[i][j] = 1e-9
            else:
                heu[i][j] += 1e-9
    
    aco = ACO(dist_mat, heu, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)
    return obj

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
        num_instances = 16  # Use fewer instances for faster testing
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
        dataset_dir = path.join(dataset_dir, "val")        # Validation mode - test on different problem sizes
        for test_size in [10, 20, 50, 100]:
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
                    print(f"Solving instance {i} of size {test_size}")
                    obj = solve(dist_mat)
                    objs.append(obj)
                else:
                    print(f"[*] Warning: Instance file {instance_file} not found")
            
            if objs:
                print(f"[*] Average for {test_size}: {calculate_mean(objs)}")
            else:
                print(f"[*] No valid instances found for size {test_size}")