from os import path
import sys
import random
import inspect
import gpt
from algo import GeneticAlgorithmTSP

def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name

possible_func_names = ["mutate", "mutate_v1", "mutate_v2", "mutate_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
mutate_func = getattr(gpt, heuristic_name)

def solve(dist_mat):
    """Solve TSP using GA with evolved mutate function."""
    # Replace the mutate function in the module with the evolved one
    import algo
    algo.mutate = mutate_func
    
    ga = GeneticAlgorithmTSP()
    ga.load_problem_from_matrix(dist_mat)
    
    # Set random seed for reproducibility
    random.seed(42)
    # Run the genetic algorithm
    best_distance = ga.run()
    return best_distance

def calculate_mean(values):
    """Calculate mean without numpy."""
    if not values:
        return 0.0
    return sum(values) / len(values)

if __name__ == "__main__":
    print("[*] Running TSP GA evaluation...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']

    # Use the existing TSP dataset
    dataset_dir = path.join(path.dirname(__file__), "../../tsp_dataset")
    
    if mood == 'train':
        # Load training instances
        objs = []
        num_instances = 16  # Use 5 instances for training
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
        # Validation mode - test on different problem sizes
        with open(path.join(path.dirname(__file__), "validation.txt"), 'w') as validation_file:
            for test_size in [10, 20, 50, 100, 200]:
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
                    validation_file.write(f"Average for {test_size}: {calculate_mean(objs)}\n")
                    print(f"[*] Average for {test_size}: {calculate_mean(objs)}")