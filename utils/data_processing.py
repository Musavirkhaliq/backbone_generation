import numpy as np
from biopandas.pdb import PandasPdb

def load_pdb(pdb_path):
    """Load a PDB file and extract Cα coordinates."""
    ppdb = PandasPdb().read_pdb(pdb_path)
    ca_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
    ca_coords = ca_atoms[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    return ca_coords

def save_coords(coords, output_path):
    """Save coordinates to a file."""
    np.save(output_path, coords)

def process_pdb_directory(input_dir, output_dir):
    """Process all PDB files in a directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    for pdb_file in os.listdir(input_dir):
        if pdb_file.endswith('.pdb'):
            pdb_path = os.path.join(input_dir, pdb_file)
            coords = load_pdb(pdb_path)
            save_path = os.path.join(output_dir, pdb_file.replace('.pdb', '.npy'))
            save_coords(coords, save_path)