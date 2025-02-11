import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class EGNNLayer(MessagePassing):
    """Equivariant GNN layer."""
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='mean')
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(node_dim, 1),
            nn.Tanh()
        )

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        edge_feat = torch.norm(pos_i - pos_j, dim=1, keepdim=True)
        return self.node_mlp(torch.cat([x_i, edge_feat], dim=-1))

    def update(self, aggr_out, pos):
        coord_update = self.coord_mlp(aggr_out)
        return aggr_out, pos + coord_update

class EGNN(nn.Module):
    """Equivariant GNN model."""
    def __init__(self, node_dim, edge_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EGNNLayer(node_dim, edge_dim) for _ in range(num_layers)])

    def forward(self, x, pos, edge_index):
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index)
        return x, pos