# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import argparse

# def plot_protein_backbone(coords, output_path=None, title="Protein Backbone"):
#     """
#     Plot a 3D visualization of a protein backbone from Cα coordinates.
    
#     Args:
#         coords: NumPy array of shape [num_residues, 3] containing Cα coordinates
#         output_path: Path to save the figure (if None, the figure is displayed)
#         title: Title for the plot
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Plot the backbone as a line
#     ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'b-', linewidth=2, alpha=0.7)
    
#     # Plot the Cα atoms as points
#     ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='r', s=50, alpha=0.8)
    
#     # Set labels and title
#     ax.set_xlabel('X (Å)')
#     ax.set_ylabel('Y (Å)')
#     ax.set_zlabel('Z (Å)')
#     ax.set_title(title)
    
#     # Set equal aspect ratio
#     max_range = np.array([
#         coords[:, 0].max() - coords[:, 0].min(),
#         coords[:, 1].max() - coords[:, 1].min(),
#         coords[:, 2].max() - coords[:, 2].min()
#     ]).max() / 2.0
    
#     mid_x = (coords[:, 0].max() + coords[:, 0].min()) / 2
#     mid_y = (coords[:, 1].max() + coords[:, 1].min()) / 2
#     mid_z = (coords[:, 2].max() + coords[:, 2].min()) / 2
    
#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"Figure saved to {output_path}")
#     else:
#         plt.show()
    
#     plt.close()

# def main(args):
#     # Load coordinates
#     coords = np.load(args.input_file)
    
#     # Create output directory if needed
#     if args.output_file:
#         os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
#     # Plot the protein backbone
#     plot_protein_backbone(
#         coords, 
#         output_path=args.output_file,
#         title=args.title
#     )

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Visualize protein backbone structures")
#     parser.add_argument("--input_file", type=str, required=True, help="Path to the .npy file containing coordinates")
#     parser.add_argument("--output_file", type=str, default=None, help="Path to save the figure (if not provided, the figure is displayed)")
#     parser.add_argument("--title", type=str, default="Protein Backbone", help="Title for the plot")
    
#     args = parser.parse_args()
#     main(args) 



