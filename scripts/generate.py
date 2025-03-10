import os
import numpy as np
import torch
from torch_geometric.data import Data
import argparse
from models.diffusion_model import DiffusionModel

def main(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = DiffusionModel(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        timesteps=args.timesteps,
        schedule='cosine',
        use_time_embedding=True
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {args.checkpoint_path}")
    
    # Create a template graph structure
    num_nodes = args.num_residues
    
    # Create node features
    x = torch.zeros(num_nodes, args.node_dim).to(device)
    
    # Create a fully connected graph
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Exclude self-loops
                edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
    
    # Initialize edge features (could be based on sequence information)
    edge_attr = torch.ones(edge_index.shape[1], args.edge_dim).to(device)
    
    print(f"Generating protein with {num_nodes} residues...")
    
    # Generate protein structure
    with torch.no_grad():
        generated_pos = model.sample(
            x,
            edge_index,
            edge_attr,
            num_steps=args.sampling_steps,
            temperature=args.temperature,
            return_trajectory=args.save_trajectory
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the generated structure
    if args.save_trajectory:
        # Save the entire trajectory
        for i, pos in enumerate(generated_pos):
            np.save(f"{args.output_dir}/generated_step_{i}.npy", pos.cpu().numpy())
        print(f"Saved generation trajectory with {len(generated_pos)} steps to {args.output_dir}")
    else:
        # Save just the final structure
        np.save(f"{args.output_dir}/generated_protein.npy", generated_pos.cpu().numpy())
        print(f"Saved generated protein to {args.output_dir}/generated_protein.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein structures using a diffusion model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/generated", help="Directory to save generated structures")
    parser.add_argument("--num_residues", type=int, default=100, help="Number of residues in the generated protein")
    parser.add_argument("--sampling_steps", type=int, default=1000, help="Number of sampling steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (higher = more diverse)")
    parser.add_argument("--save_trajectory", action="store_true", help="Save the entire generation trajectory")
    parser.add_argument("--node_dim", type=int, default=16, help="Node feature dimension")
    parser.add_argument("--edge_dim", type=int, default=1, help="Edge feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of EGNN layers")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    
    args = parser.parse_args()
    main(args)