# This file is no longer needed as we're using the tsp_dataset directly
# The original dataset generation has been replaced with direct loading from tsp_dataset

def generate_datasets():
    """
    This function is no longer needed but kept for backward compatibility.
    The TSP ACO solver now loads datasets directly from ../../tsp_dataset/
    """
    print("Dataset generation is no longer needed. Using existing tsp_dataset.")
    pass

if __name__ == "__main__":
    generate_datasets()