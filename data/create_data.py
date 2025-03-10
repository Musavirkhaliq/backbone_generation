import numpy as np
from Bio.PDB import PDBParser, PDBList
from Bio.PDB.vectors import calc_angle, calc_dihedral
import os

# Function to download PDB files
def download_pdb_subset(pdb_ids, download_dir="pdb_files"):
    pdbl = PDBList()
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    for pdb_id in pdb_ids:
        pdbl.retrieve_pdb_file(pdb_id, pdir=download_dir, file_format="pdb")
    return [os.path.join(download_dir, f"pdb{pdb_id.lower()}.ent") for pdb_id in pdb_ids]

# Function to extract backbone coordinates (N, Cα, C) from a PDB file
def extract_backbone_coords(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    coords = []
    atom_order = ["N", "CA", "C"]  # N, Cα, C for backbone
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_coords = []
                for atom_name in atom_order:
                    if atom_name in residue:
                        atom = residue[atom_name]
                        residue_coords.append(atom.get_coord())
                if len(residue_coords) == 3:  # Ensure all N, CA, C are present
                    coords.extend(residue_coords)
            break  # Use first chain for simplicity
        break  # Use first model
    return np.array(coords)  # Shape: [3N, 3] for N residues

# Function to compute distances between consecutive atoms
def compute_distances(coords):
    distances = []
    for i in range(len(coords) - 1):
        vec = coords[i + 1] - coords[i]
        dist = np.linalg.norm(vec)
        distances.append(dist)
    return np.array(distances)  # Shape: [3N-1]

# Function to compute bond angles
def compute_bond_angles(coords):
    angles = []
    for i in range(len(coords) - 2):
        v1 = coords[i] - coords[i + 1]  # Vector from i+1 to i
        v2 = coords[i + 2] - coords[i + 1]  # Vector from i+1 to i+2
        angle = calc_angle(coords[i], coords[i + 1], coords[i + 2])  # In radians
        angles.append(np.degrees(angle))  # Convert to degrees
    return np.array(angles)  # Shape: [3N-2]

# Function to compute torsion angles (φ, ψ, ω)
def compute_torsion_angles(coords):
    torsions = []
    n_residues = len(coords) // 3
    for i in range(n_residues):
        if i > 0:  # φ: C(i-1)-N(i)-Cα(i)-C(i)
            torsion = calc_dihedral(coords[3*(i-1) + 2], coords[3*i], coords[3*i + 1], coords[3*i + 2])
            torsions.append(np.degrees(torsion))
        if i < n_residues - 1:  # ψ: N(i)-Cα(i)-C(i)-N(i+1)
            torsion = calc_dihedral(coords[3*i], coords[3*i + 1], coords[3*i + 2], coords[3*(i+1)])
            torsions.append(np.degrees(torsion))
        if i < n_residues - 1:  # ω: Cα(i)-C(i)-N(i+1)-Cα(i+1)
            torsion = calc_dihedral(coords[3*i + 1], coords[3*i + 2], coords[3*(i+1)], coords[3*(i+1) + 1])
            torsions.append(np.degrees(torsion))
    return np.array(torsions)

# Preprocessing pipeline
def preprocess_protein(pdb_file):
    # Extract coordinates
    coords = extract_backbone_coords(pdb_file)
    if len(coords) == 0:
        raise ValueError("No valid backbone atoms found in PDB file.")
    
    # # Compute distances and angles
    # distances = compute_distances(coords)
    # bond_angles = compute_bond_angles(coords)
    # torsion_angles = compute_torsion_angles(coords)
    
    # Combined representation
    data = {
        "coords": coords,              # [3N, 3]
        # "distances": distances,        # [3N-1]
        # "bond_angles": bond_angles,    # [3N-2]
        # "torsion_angles": torsion_angles  # Variable length, ~3N for φ, ψ, ω
    }
    return data

# Main execution
if __name__ == "__main__":
    # Example: Download a small subset of PDB files (short proteins/peptides)
    pdb_ids = ["1A1E", "1BTA"]  # Small proteins for demo
    pdb_files = download_pdb_subset(pdb_ids)
    
    # Preprocess each PDB file
    for pdb_file in pdb_files:
        try:
            data = preprocess_protein(pdb_file)
            print(f"Processed {pdb_file}:")
            print(f"  Coordinates shape: {data['coords'].shape}")
            # print(f"  Distances: {data['distances'][:5]}... (len={len(data['distances'])})")
            # print(f"  Bond angles: {data['bond_angles'][:5]}... (len={len(data['bond_angles'])})")
            # print(f"  Torsion angles: {data['torsion_angles'][:5]}... (len={len(data['torsion_angles'])})")
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

    # Save to file (optional)
    # np.savez("protein_data.npz", **data)




## Data 

Coordinated of extract_backbone_coords