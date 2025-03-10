import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from typing import Optional, Tuple
from torch_scatter import scatter

class EGNNLayer(MessagePassing):
    """
    Equivariant Graph Neural Network Layer.
    
    Implements E(n) equivariant message passing with:
    - Coordinate normalization
    - Edge attributes support
    - Residual connections
    - Layer normalization
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        activation: nn.Module = nn.SiLU(),
        aggr: str = 'mean',
        use_layer_norm: bool = True
    ):
        # Important: Set node_dim=0 for MessagePassing
        super().__init__(aggr=aggr, node_dim=0)
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Message MLPs
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 1, hidden_dim),  # +1 for distance
            activation,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Coordinate update network
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(node_dim) if use_layer_norm else None
        
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the layer.
        
        Args:
            x: Node features [num_nodes, node_dim]
            pos: Node positions [num_nodes, 3]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Updated node features and positions
        """
        # Normalize coordinates to prevent numerical issues
        pos_mean = pos.mean(dim=0, keepdim=True)
        pos_std = pos.std(dim=0, keepdim=True).clamp(min=1e-8)
        pos_normalized = (pos - pos_mean) / pos_std
        
        # Compute messages and updates for node features
        out = self.propagate(edge_index, x=x, pos=pos_normalized, edge_attr=edge_attr, update_coords=False)
        
        # Update node features with residual connection
        if self.layer_norm is not None:
            out = self.layer_norm(out + x)
        else:
            out = out + x
            
        # Update coordinates
        pos_update = self.propagate(
            edge_index,
            x=out,
            pos=pos_normalized,
            edge_attr=edge_attr,
            update_coords=True
        )
        
        # Denormalize coordinate updates
        new_pos = pos + pos_update * pos_std
        
        return out, new_pos
        
    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        edge_attr: OptTensor,
        update_coords: bool = False
    ) -> Tensor:
        """Compute messages between nodes."""
        # Compute relative positions and distances
        rel_pos = pos_i - pos_j
        dist = torch.norm(rel_pos, dim=1, keepdim=True)
        
        if update_coords:
            # For coordinate updates
            message = self.coord_mlp(x_i)
            return message.view(-1, 1) * rel_pos
            
        # Compute edge features
        if edge_attr is None:
            edge_features = dist
        else:
            edge_features = torch.cat([edge_attr, dist], dim=-1)
            
        # Process edge features
        edge_features = self.edge_mlp(edge_features)
        
        # Combine node and edge features
        message = torch.cat([x_i, x_j, edge_features], dim=-1)
        message = self.node_mlp(message)
        
        return message

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        update_coords: bool = False
    ) -> Tensor:
        """Aggregate messages from neighbors."""
        # Use different aggregation for coordinate updates
        if update_coords:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)

class EGNN(nn.Module):
    """
    Enhanced Equivariant Graph Neural Network.
    
    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
        hidden_dim: Dimension of hidden layers
        num_layers: Number of EGNN layers
        dropout: Dropout rate
        activation: Activation function
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: nn.Module = nn.SiLU()
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            EGNNLayer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_all_outputs: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Node features [num_nodes, node_dim]
            pos: Node positions [num_nodes, 3]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            return_all_outputs: If True, return outputs from all layers
            
        Returns:
            Updated node features and positions
        """
        outputs = []
        
        for layer in self.layers:
            x = self.dropout(x)
            x, pos = layer(x, pos, edge_index, edge_attr)
            outputs.append((x, pos))
            
        if return_all_outputs:
            return outputs
        return x, pos