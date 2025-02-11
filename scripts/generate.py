import torch
from models.diffusion_model import DiffusionModel

# Initialize model
model = DiffusionModel(node_dim=16, edge_dim=1, num_layers=4, timesteps=1000)
model.load_state_dict(torch.load('outputs/trained_models/model.pth'))

# Generate
x = torch.randn(100, 16)  # Random node features
pos = torch.randn(100, 3)  # Random initial positions
edge_index = torch.randint(0, 100, (2, 500))  # Random edges

for t in range(1000, 0, -1):
    pos = model(x, pos, edge_index, t)

# Save generated structure
torch.save(pos, 'outputs/generated_structures/generated.pth')