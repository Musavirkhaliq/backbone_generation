import os
import numpy as np
from biopandas.pdb import PandasPdb
import requests

# Directory setup
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# List of PDB IDs to download (small dataset example)
PDB_IDS = ["1crn", "1alm", "1ubq", "1hhp", "1mbn"]  # Add more IDs as needed

def download_pdb(pdb_id, save_dir):
    """Download a PDB file from the RCSB PDB database."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(save_dir, f"{pdb_id}.pdb"), "wb") as f:
            f.write(response.content)
        print(f"Downloaded {pdb_id}.pdb")
    else:
        print(f"Failed to download {pdb_id}.pdb")

def extract_ca_coords(pdb_path):
    """Extract Cα coordinates from a PDB file."""
    ppdb = PandasPdb().read_pdb(pdb_path)
    ca_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
    ca_coords = ca_atoms[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    return ca_coords

def save_coords(coords, output_path):
    """Save coordinates to a file."""
    np.save(output_path, coords)

def preprocess_pdb(pdb_id, raw_dir, processed_dir):
    """Preprocess a PDB file into Cα coordinates."""
    pdb_path = os.path.join(raw_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        print(f"PDB file {pdb_id}.pdb not found.")
        return

    # Extract Cα coordinates
    coords = extract_ca_coords(pdb_path)
    if coords.size == 0:
        print(f"No Cα atoms found in {pdb_id}.pdb")
        return

    # Save processed coordinates
    save_path = os.path.join(processed_dir, f"{pdb_id}.npy")
    save_coords(coords, save_path)
    print(f"Processed and saved {pdb_id}.npy")

def main():
    # Download PDB files
    for pdb_id in PDB_IDS:
        download_pdb(pdb_id, RAW_DIR)

    # Preprocess PDB files
    for pdb_id in PDB_IDS:
        preprocess_pdb(pdb_id, RAW_DIR, PROCESSED_DIR)

if __name__ == "__main__":
    main()