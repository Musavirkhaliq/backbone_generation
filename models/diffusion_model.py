import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class NoiseScheduler:
    """
    Noise scheduler for diffusion process with support for various schedules.
    
    Args:
        timesteps: Number of diffusion timesteps
        schedule: Type of beta schedule ('linear', 'cosine', or 'quadratic')
        beta_start: Starting beta value
        beta_end: Ending beta value
    """
    def __init__(
        self,
        timesteps: int,
        schedule: str = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        self.timesteps = timesteps
        
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == 'cosine':
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / (1 + 0.008) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        elif schedule == 'quadratic':
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
        
        # Compute diffusion parameters
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0) for a batch of timesteps."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

class DiffusionModel(nn.Module):
    """
    Enhanced diffusion model for protein backbone generation.
    
    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
        hidden_dim: Hidden dimension size
        num_layers: Number of EGNN layers
        timesteps: Number of diffusion timesteps
        schedule: Type of noise schedule
        noise_scale: Scale factor for noise prediction
        use_time_embedding: Whether to use learnable time embeddings
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        timesteps: int = 1000,
        schedule: str = 'cosine',
        noise_scale: float = 1.0,
        use_time_embedding: bool = True
    ):
        super().__init__()
        
        self.timesteps = timesteps
        self.noise_scale = noise_scale
        self.use_time_embedding = use_time_embedding
        
        # Time embedding
        if use_time_embedding:
            self.time_embed_dim = hidden_dim
            self.time_embed = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # EGNN backbone
        self.egnn = EGNN(
            node_dim=node_dim + (hidden_dim if use_time_embedding else 0),
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(timesteps, schedule)
        
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Create time embeddings."""
        half_dim = self.time_embed_dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return self.time_embed(embeddings)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass predicting noise in the diffusion process.
        
        Args:
            x: Node features [num_nodes, node_dim]
            pos: Node positions [num_nodes, 3]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            t: Timesteps [batch_size]
            
        Returns:
            Predicted noise and noisy positions
        """
        # Sample timesteps if not provided
        if t is None:
            t = torch.randint(0, self.timesteps, (pos.shape[0],), device=pos.device)
            
        # Add noise according to schedule
        noise = torch.randn_like(pos) * self.noise_scale
        noisy_pos = self.noise_scheduler.q_sample(pos, t, noise)
        
        # Prepare time embeddings
        if self.use_time_embedding:
            time_emb = self.get_time_embedding(t)
            x = torch.cat([x, time_emb.repeat_interleave(x.size(0) // t.size(0), 0)], dim=-1)
        
        # Predict noise using EGNN
        _, pred_noise = self.egnn(x, noisy_pos, edge_index, edge_attr)
        
        return pred_noise, noisy_pos
    
    def sample(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Generate samples using DDPM sampling.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            num_steps: Number of sampling steps (default: self.timesteps)
            temperature: Sampling temperature
            return_trajectory: Whether to return the entire sampling trajectory
            
        Returns:
            Generated positions [num_nodes, 3]
        """
        device = x.device
        num_steps = num_steps or self.timesteps
        trajectory = []
        
        # Start from pure noise
        pos = torch.randn(x.size(0), 3, device=device) * temperature
        
        # Gradually denoise
        for t in reversed(range(num_steps)):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            
            # Get noise prediction
            with torch.no_grad():
                noise_pred, _ = self.forward(x, pos, edge_index, edge_attr, t_batch)
            
            # Get posterior parameters
            alpha = self.noise_scheduler.alphas[t]
            alpha_prev = self.noise_scheduler.alphas_cumprod_prev[t]
            beta = self.noise_scheduler.betas[t]
            
            # Update position estimate
            pred_mean = (pos - beta * noise_pred / torch.sqrt(1 - alpha)) / torch.sqrt(alpha)
            posterior_variance = beta * (1 - alpha_prev) / (1 - alpha)
            
            pos = pred_mean + torch.randn_like(pos) * torch.sqrt(posterior_variance) * temperature
            
            if return_trajectory:
                trajectory.append(pos.clone())
        
        if return_trajectory:
            return torch.stack(trajectory)
        return pos

    def loss_fn(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute the diffusion loss for training.
        
        Args:
            x: Node features [num_nodes, node_dim]
            pos: Node positions [num_nodes, 3]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            reduction: Loss reduction method ('mean' or 'none')
            
        Returns:
            Loss value
        """
        # Forward pass with random timesteps
        pred_noise, noisy_pos = self.forward(x, pos, edge_index, edge_attr)
        
        # Compute MSE loss
        loss = F.mse_loss(pred_noise, pos - noisy_pos, reduction=reduction)
        
        return loss