import os

def create_directory_structure(root_dir):
    """
    Creates the directory structure for the protein diffusion project.

    Args:
        root_dir (str): The root directory of the project.
    """

    # Create the root directory if it doesn't exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Create the subdirectories
    subdirs = [
        "data",
        "data/raw",
        "data/processed",
        "data/splits",
        "models",
        "utils",
        "scripts",
        "configs",
        "outputs",
        "outputs/trained_models",
        "outputs/generated_structures",
        "outputs/logs",
    ]

    for subdir in subdirs:
        full_path = os.path.join(root_dir, subdir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

    # Create files
    files = [
        "models/diffusion_model.py",
        "models/en_gnn.py",
        "utils/data_processing.py",
        "utils/evaluation.py",
        "utils/visualization.py",
        "scripts/train.py",
        "scripts/generate.py",
        "scripts/evaluate.py",
        "configs/config.yaml",
        "requirements.txt",
    ]

    for file in files:
        full_path = os.path.join(root_dir, file)
        if not os.path.exists(full_path):
            with open(full_path, "w") as f:
                pass  # Create an empty file

# Example usage
root_dir = "."
create_directory_structure(root_dir)