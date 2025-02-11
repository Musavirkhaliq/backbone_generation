import torch
import torch.optim as optim
from models.diffusion_model import DiffusionModel
from utils.data_processing import load_coords

# Load data
coords = load_coords('data/processed/example.npy')
x = torch.randn(len(coords), 16)  # Node features
pos = torch.tensor(coords, dtype=torch.float32)
edge_index = torch.randint(0, len(coords), (2, 100))  # Random edges

# Initialize model
model = DiffusionModel(node_dim=16, edge_dim=1, num_layers=4, timesteps=1000)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(100):
    t = torch.randint(0, 1000, (1,))
    denoised_pos = model(x, pos, edge_index, t)
    loss = torch.nn.functional.mse_loss(denoised_pos, pos)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")