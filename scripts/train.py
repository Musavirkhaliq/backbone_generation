import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from models.diffusion_model import DiffusionModel

# Configuration
DATA_DIR = "data/processed"
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
TIMESTEPS = 1000
NODE_DIM = 16
EDGE_DIM = 1
HIDDEN_DIM = 128
NUM_LAYERS = 4
SAVE_INTERVAL = 10

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        print(f"Found {len(self.file_list)} protein structures")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        coords = np.load(file_path)
        
        # Create node features (could be amino acid properties in a real scenario)
        num_nodes = coords.shape[0]
        
        # Create a fully connected graph for simplicity
        # In a more sophisticated implementation, you might use distance-based cutoffs
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Calculate edge features (distances between Cα atoms)
        edge_attr = torch.zeros(edge_index.shape[1], EDGE_DIM)
        for e in range(edge_index.shape[1]):
            i, j = edge_index[0, e], edge_index[1, e]
            dist = np.linalg.norm(coords[i] - coords[j])
            edge_attr[e, 0] = dist
        
        # Create a Data object
        x = torch.zeros(num_nodes, NODE_DIM)  # Initialize with zeros or one-hot encoding of amino acids
        pos = torch.tensor(coords, dtype=torch.float32)
        
        data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        
        return data

# Load dataset
dataset = ProteinDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = DiffusionModel(
    node_dim=NODE_DIM,
    edge_dim=EDGE_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    timesteps=TIMESTEPS,
    schedule='cosine',
    use_time_embedding=True
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Create output directories
os.makedirs("outputs/trained_models", exist_ok=True)
os.makedirs("outputs/samples", exist_ok=True)

# Training loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Compute loss using the model's loss function
        loss = model.loss_fn(batch.x, batch.pos, batch.edge_index, batch.edge_attr)
        
        loss.backward()
        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")
    
    # Update learning rate
    scheduler.step(avg_loss)
    
    # Save model checkpoint
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"outputs/trained_models/diffusion_model_epoch_{epoch+1}.pth")
        
        # Generate and save a sample
        if len(dataset) > 0:
            model.eval()
            with torch.no_grad():
                # Get a sample from the dataset for conditioning
                sample_data = dataset[0].to(device)
                
                # Generate a new protein structure
                generated_pos = model.sample(
                    sample_data.x,
                    sample_data.edge_index,
                    sample_data.edge_attr,
                    num_steps=100,  # Use fewer steps for faster generation during training
                    temperature=0.8
                )
                
                # Save the generated structure
                np.save(f"outputs/samples/generated_epoch_{epoch+1}.npy", generated_pos.cpu().numpy())

# Save the final model
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, "outputs/trained_models/diffusion_model_final.pth")

print("Training complete. Model saved.")