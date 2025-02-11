import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.diffusion_model import DiffusionModel
from models.en_gnn import EGNN

# Configuration
DATA_DIR = "data/processed"
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
TIMESTEPS = 1000
NODE_DIM = 16
EDGE_DIM = 1
NUM_LAYERS = 4

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        coords = np.load(file_path)
        return torch.tensor(coords, dtype=torch.float32)

# Collate function for DataLoader
def collate_fn(batch):
    coords = torch.stack(batch)
    num_nodes = coords.shape[1]
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4))  # Random edges
    return coords, edge_index

# Load dataset
dataset = ProteinDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Initialize model
model = DiffusionModel(node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_layers=NUM_LAYERS, timesteps=TIMESTEPS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Noise schedule
beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (coords, edge_index) in enumerate(dataloader):
        coords, edge_index = coords.to(device), edge_index.to(device)
        batch_size, num_nodes, _ = coords.shape

        # Random timestep
        t = torch.randint(0, TIMESTEPS, (batch_size,), device=device)

        # Add noise to coordinates
        noise = torch.randn_like(coords)
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1)
        noisy_coords = sqrt_alpha_bar * coords + sqrt_one_minus_alpha_bar * noise

        # Predict denoised coordinates
        x = torch.randn(batch_size, num_nodes, NODE_DIM, device=device)  # Random node features
        denoised_coords = model(x, noisy_coords, edge_index, t)

        # Compute loss
        loss = nn.functional.mse_loss(denoised_coords, coords)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

# Save the trained model
os.makedirs("outputs/trained_models", exist_ok=True)
torch.save(model.state_dict(), "outputs/trained_models/diffusion_model.pth")
print("Training complete. Model saved.")