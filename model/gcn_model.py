import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from torch.nn import BatchNorm1d, Dropout, ReLU, Linear

class ImprovedGCN(torch.nn.Module):
    def __init__(
        self, 
        num_node_features: int,
        hidden_channels: int | list,
        num_classes: int = 1,
        dropout_rate: float = 0.3,
        pool_type: str = 'mean',
        residual: bool = True,
        seed: int = 42
    ):
        """
        Initialize the Improved GCN model.
        
        Args:
            num_node_features (int): Number of input node features
            hidden_channels (int or list): Hidden channel dimensions. If int, creates two layers
                                         with same number of channels. If list, creates layers
                                         with specified channels.
            num_classes (int): Number of output classes (default: 1 for binary classification)
            dropout_rate (float): Dropout rate (default: 0.3)
            pool_type (str): Type of global pooling ('mean', 'max', or 'add')
            residual (bool): Whether to use residual connections
            seed (int): Random seed for reproducibility
        """
        super(ImprovedGCN, self).__init__()
        torch.manual_seed(seed)
        
        # Convert hidden_channels to list if it's an integer
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels, hidden_channels]
        
        # Model components
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.residual = residual
        
        # Input layer
        self.convs.append(GCNConv(num_node_features, hidden_channels[0]))
        self.batch_norms.append(BatchNorm1d(hidden_channels[0]))
        
        # Hidden layers
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i + 1]))
            self.batch_norms.append(BatchNorm1d(hidden_channels[i + 1]))
        
        # Output layers
        self.dropout = Dropout(p=dropout_rate)
        self.final_lin1 = Linear(hidden_channels[-1], hidden_channels[-1] // 2)
        self.final_lin2 = Linear(hidden_channels[-1] // 2, num_classes)
        
        # Pooling function
        if pool_type == 'max':
            self.pool = global_max_pool
        elif pool_type == 'add':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Node features [num_nodes, num_node_features]
            edge_index (Tensor): Graph connectivity [2, num_edges]
            batch (Tensor): Batch vector [num_nodes]
            
        Returns:
            Tensor: Predictions
        """
        # Initial features
        previous = None
        
        # Graph convolution layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Store previous output for residual connection
            if i > 0 and self.residual:
                previous = x
                
            # Apply convolution
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Add residual connection if dimensions match
            if i > 0 and self.residual and x.size(-1) == previous.size(-1):
                x = x + previous
        
        # Global pooling
        x = self.pool(x, batch)
        
        # MLP head
        x = self.dropout(x)
        x = F.relu(self.final_lin1(x))
        x = self.dropout(x)
        x = self.final_lin2(x)
        
        # Output activation based on number of classes
        if self.final_lin2.out_features == 1:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=-1)
    
    def reset_parameters(self):
        """Reset all learnable parameters of the model."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        self.final_lin1.reset_parameters()
        self.final_lin2.reset_parameters()