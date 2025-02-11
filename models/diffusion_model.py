import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    """Diffusion model for protein backbone generation."""
    def __init__(self, node_dim, edge_dim, num_layers, timesteps):
        super().__init__()
        self.egnn = EGNN(node_dim, edge_dim, num_layers)
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps)

    def forward(self, x, pos, edge_index, t):
        noise = torch.randn_like(pos)
        alpha = 1 - self.beta[t]
        noisy_pos = torch.sqrt(alpha) * pos + torch.sqrt(1 - alpha) * noise
        _, denoised_pos = self.egnn(x, noisy_pos, edge_index)
        return denoised_pos