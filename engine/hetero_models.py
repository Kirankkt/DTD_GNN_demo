"""
Heterogeneous GNN Models for DTD-GNN Full Architecture

This module implements TransformerConv-based models that work with
the heterogeneous Event Node graph structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, HeteroConv, Linear


class HeteroTransformerGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network using TransformerConv.
    
    Designed for the DTD-GNN Event Node structure with bidirectional edges.
    """
    
    def __init__(self, in_channels=1024, hidden_channels=256, out_channels=128, num_heads=8):
        super().__init__()
        
        # First layer: Heterogeneous convolution with bidirectional edges
        self.conv1 = HeteroConv({
            # Forward edges
            ('drug', 'participates', 'event'): TransformerConv(
                in_channels, hidden_channels // num_heads, heads=num_heads, dropout=0.1
            ),
            ('target', 'participates', 'event'): TransformerConv(
                in_channels, hidden_channels // num_heads, heads=num_heads, dropout=0.1
            ),
            ('event', 'treats', 'disease'): TransformerConv(
                in_channels, hidden_channels // num_heads, heads=num_heads, dropout=0.1
            ),
            # Reverse edges (so all node types get updated)
            ('event', 'rev_participates', 'drug'): TransformerConv(
                in_channels, hidden_channels // num_heads, heads=num_heads, dropout=0.1
            ),
            ('event', 'rev_participates', 'target'): TransformerConv(
                in_channels, hidden_channels // num_heads, heads=num_heads, dropout=0.1
            ),
            ('disease', 'rev_treats', 'event'): TransformerConv(
                in_channels, hidden_channels // num_heads, heads=num_heads, dropout=0.1
            ),
        }, aggr='sum')
        
        # Layer normalization per node type
        self.norms1 = nn.ModuleDict({
            'drug': nn.LayerNorm(hidden_channels),
            'target': nn.LayerNorm(hidden_channels),
            'disease': nn.LayerNorm(hidden_channels),
            'event': nn.LayerNorm(hidden_channels),
        })
        
        # Second layer
        self.conv2 = HeteroConv({
            ('drug', 'participates', 'event'): TransformerConv(
                hidden_channels, out_channels // num_heads, heads=num_heads, dropout=0.1
            ),
            ('target', 'participates', 'event'): TransformerConv(
                hidden_channels, out_channels // num_heads, heads=num_heads, dropout=0.1
            ),
            ('event', 'treats', 'disease'): TransformerConv(
                hidden_channels, out_channels // num_heads, heads=num_heads, dropout=0.1
            ),
            ('event', 'rev_participates', 'drug'): TransformerConv(
                hidden_channels, out_channels // num_heads, heads=num_heads, dropout=0.1
            ),
            ('event', 'rev_participates', 'target'): TransformerConv(
                hidden_channels, out_channels // num_heads, heads=num_heads, dropout=0.1
            ),
            ('disease', 'rev_treats', 'event'): TransformerConv(
                hidden_channels, out_channels // num_heads, heads=num_heads, dropout=0.1
            ),
        }, aggr='sum')
        
        self.norms2 = nn.ModuleDict({
            'drug': nn.LayerNorm(out_channels),
            'target': nn.LayerNorm(out_channels),
            'disease': nn.LayerNorm(out_channels),
            'event': nn.LayerNorm(out_channels),
        })
        
        # Linear projections for skip connections
        self.lin_skip = nn.ModuleDict({
            'drug': nn.Linear(in_channels, hidden_channels),
            'target': nn.Linear(in_channels, hidden_channels),
            'disease': nn.Linear(in_channels, hidden_channels),
            'event': nn.Linear(in_channels, hidden_channels),
        })
        
        # Second layer skip projections
        self.lin_skip2 = nn.ModuleDict({
            'drug': nn.Linear(hidden_channels, out_channels),
            'target': nn.Linear(hidden_channels, out_channels),
            'disease': nn.Linear(hidden_channels, out_channels),
            'event': nn.Linear(hidden_channels, out_channels),
        })
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the heterogeneous GNN.
        
        Args:
            x_dict: Dict of node features {node_type: tensor}
            edge_index_dict: Dict of edge indices {edge_type: tensor}
        
        Returns:
            Dict of node embeddings {node_type: tensor}
        """
        # Skip connections
        skip = {key: self.lin_skip[key](x) for key, x in x_dict.items()}
        
        # First conv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        
        # Add skip + normalize + activate
        for key in x_dict:
            if key in skip:
                x_dict[key] = self.norms1[key](x_dict[key] + skip[key])
                x_dict[key] = F.elu(x_dict[key])
        
        # Second conv layer
        skip2 = {key: self.lin_skip2[key](x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        for key in x_dict:
            if key in skip2:
                x_dict[key] = self.norms2[key](x_dict[key] + skip2[key])
                x_dict[key] = F.elu(x_dict[key])
        
        return x_dict


class HeteroLinkPredictor(nn.Module):
    """
    Link predictor for heterogeneous graphs.
    Predicts the probability of an edge between event and disease nodes.
    """
    
    def __init__(self, in_channels=128):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 2, in_channels)
        self.lin2 = nn.Linear(in_channels, 1)
    
    def forward(self, z_event, z_disease):
        """
        Predict link probability between event and disease.
        
        Args:
            z_event: Event node embeddings
            z_disease: Disease node embeddings
        
        Returns:
            Probability scores
        """
        z = torch.cat([z_event, z_disease], dim=-1)
        z = F.relu(self.lin1(z))
        return torch.sigmoid(self.lin2(z)).squeeze(-1)
