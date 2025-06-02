"""
Generate TSP test cases with the specified format:
- First line: number of cities
- Next n lines: distance matrix (symmetric, diagonal = 10^9, costs 1-1000)
- Output format: {size}_{num:02d}.inp
"""

import os
import numpy as np
import random

def generate_tsp_instance(size):
    """
    Generate a single TSP instance.
    
    Args:
        size (int): Number of cities
    
    Returns:
        numpy.ndarray: Symmetric distance matrix
    """
    
    # Generate upper triangular matrix with random costs (1-1000)
    matrix = np.zeros((size, size), dtype=float)
    
    # Fill upper triangle with random values
    for i in range(size):
        for j in range(i + 1, size):
            cost = random.random() * 999 + 1  # Random float in range [1, 1000)
            # make cost 3 decimal places
            cost = round(cost, 3)
            matrix[i][j] = cost
            matrix[j][i] = cost  # Make symmetric
    
    # Set diagonal to large number (10^9)
    for i in range(size):
        matrix[i][i] = 10**9
    
    return matrix

def save_tsp_instance(matrix, filepath):
    """
    Save TSP instance to file in the specified format.
    
    Args:
        matrix (numpy.ndarray): Distance matrix
        filepath (str): Output file path
    """
    size = matrix.shape[0]
    
    with open(filepath, 'w') as f:
        # First line: number of cities
        f.write(f"{size}\n")
        
        # Next n lines: distance matrix
        for i in range(size):
            row = " ".join(map(str, matrix[i]))
            f.write(f"{row}\n")

def generate_tsp_datasets(mood):
    """Generate all TSP datasets according to specifications."""
    # Create output directory
    output_dir = "tsp_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Problem sizes and number of instances
    if mood == 'train':
        problem_sizes = [10, 20, 50, 100, 200]
        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        output_dir = os.path.join(output_dir, "train")
    else: 
        problem_sizes = [10, 20, 50, 100, 200]
        os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
        output_dir = os.path.join(output_dir, "val")
    n_instances = 16

    seed = 1234 + (mood == "train") * 1000
    print(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    
    total_files = 0
    
    print("Generating TSP test cases...")
    print(f"Output directory: {output_dir}")
    print(f"Problem sizes: {problem_sizes}")
    print(f"Instances per size: {n_instances}")
    print(f"Cost range: 1-1000")
    print(f"Diagonal value: 10^9")
    print(f"Matrix type: Symmetric")
    print()
    
    for size in problem_sizes:
        print(f"Generating size {size}...")
        
        for instance_num in range(n_instances):
            # Generate instance with unique seed
            matrix = generate_tsp_instance(size)
            
            # Create filename: {size}_{num:02d}.inp
            filename = f"{size}_{instance_num:02d}.inp"
            filepath = os.path.join(output_dir, filename)
            
            # Save to file
            save_tsp_instance(matrix, filepath)
            total_files += 1
            
            print(f"  Created {filename}")
    
    print()
    print(f"Dataset generation completed!")
    print(f"Total files created: {total_files}")
    print(f"File naming convention: {{size}}_{{instance:02d}}.inp")

def verify_datasets(mood):
    """Verify a few generated files to ensure correct format."""
    print("\nVerifying generated files...")
    
    output_dir = os.path.join("tsp_dataset", mood)
    test_files = ["10_00.inp", "20_00.inp", "50_00.inp"]
    
    for filename in test_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"\nChecking {filename}:")
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Check first line (dimension)
            size = int(lines[0].strip())
            print(f"  Size: {size}")
            
            # Check matrix dimensions
            matrix_lines = len(lines) - 1
            print(f"  Matrix lines: {matrix_lines}")
            
            if matrix_lines == size:
                print(f"  ✓ Correct number of matrix rows")
            else:
                print(f"  ✗ Expected {size} matrix rows, got {matrix_lines}")
            
            # Check first row for format
            first_row = lines[1].strip().split()
            print(f"  First row length: {len(first_row)}")
            
            if len(first_row) == size:
                print(f"  ✓ Correct number of columns")
            else:
                print(f"  ✗ Expected {size} columns, got {len(first_row)}")
            
            # Check diagonal value
            if len(first_row) > 0:
                diagonal_val = float(first_row[0])
                if diagonal_val == 10**9:
                    print(f"  ✓ Correct diagonal value: {diagonal_val}")
                else:
                    print(f"  ✗ Expected diagonal value 10^9, got {diagonal_val}")
            
            # Check for symmetry (sample a few elements)
            if len(lines) > 2:
                try:
                    matrix = []
                    for i in range(1, min(4, len(lines))):  # Check first few rows
                        row = list(map(float, lines[i].strip().split()))
                        matrix.append(row)
                    
                    # Check symmetry for available elements
                    symmetric = True
                    for i in range(len(matrix)):
                        for j in range(len(matrix[i])):
                            if j < len(matrix) and i < len(matrix[j]):
                                if matrix[i][j] != matrix[j][i]:
                                    symmetric = False
                                    break
                        if not symmetric:
                            break
                    
                    if symmetric:
                        print(f"  ✓ Matrix appears symmetric")
                    else:
                        print(f"  ✗ Matrix may not be symmetric")
                        
                except Exception as e:
                    print(f"  ! Error checking symmetry: {e}")

def generate_sample_with_floats():
    """Generate a few sample instances with floating-point costs for demonstration."""
    print("\nGenerating sample files with floating-point costs...")
    
    output_dir = "tsp_dataset"
    
    # Generate a few samples with float costs
    sizes = [10, 20]
    
    for size in sizes:
        # Generate matrix with float costs
        matrix = np.zeros((size, size))
        
        # Fill upper triangle with random float values (1.0 to 1000.0)
        for i in range(size):
            for j in range(i + 1, size):
                cost = np.random.uniform(1.0, 1000.0)
                matrix[i][j] = cost
                matrix[j][i] = cost  # Make symmetric
        
        # Set diagonal to large number
        for i in range(size):
            matrix[i][i] = 10**9
        
        # Save float version
        filename = f"{size}_float_sample.inp"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"{size}\n")
            for i in range(size):
                row = " ".join(f"{matrix[i][j]:.2f}" for j in range(size))
                f.write(f"{row}\n")
        
        print(f"  Created {filename} (with floating-point costs)")

if __name__ == "__main__":
    # train or val dataset generation 
    # get mood from command line arguments 
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_tsp_dataset_train.py <mood>")
        sys.exit(1)
    mood = sys.argv[1]
    generate_tsp_datasets(mood)
    verify_datasets(mood)
