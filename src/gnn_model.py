"""
GNN model for Task 2: GraphSAGE / GAT on audio segment graphs.

Architecture:
  Node features → SAGEConv/GATConv with Residuals → Dual Pooling (Mean + Max) → MLP Head → Multi-Label Logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, global_max_pool


class GraphSAGEClassifier(nn.Module):
    """
    Enhanced GraphSAGE-based multi-label classifier.

    Features:
      - Residual connections across SAGEConv layers
      - Dual pooling: concatenated global mean + max pooling for rich graph embeddings
      - LayerNorm + Dropout in MLP classification head

    Args:
        in_channels: input node feature dimension (140)
        hidden_channels: hidden layer size (128)
        num_classes: output label count (8)
        num_layers: number of SAGEConv layers (3)
        dropout: dropout rate (0.3)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.res_projs.append(
            nn.Linear(in_channels, hidden_channels, bias=False)
            if in_channels != hidden_channels
            else nn.Identity()
        )

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.res_projs.append(nn.Identity())

        # Dual pooling produces [batch_size, 2 * hidden_channels]
        pool_dim = 2 * hidden_channels
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

        self.hidden_channels = hidden_channels

    def encode(self, x, edge_index, edge_attr, batch):
        """
        Produce graph-level embedding via dual pooling (mean + max).
        Returns: [batch_size, 2 * hidden_channels]
        """
        for i, (conv, norm, res) in enumerate(zip(self.convs, self.norms, self.res_projs)):
            identity = res(x)
            h = conv(x, edge_index)
            h = norm(h)
            h = h + identity
            h = F.relu(h)
            x = F.dropout(h, p=self.dropout, training=self.training)

        # Dual pooling: average texture + peak dynamics
        mean_p = global_mean_pool(x, batch)
        max_p = global_max_pool(x, batch)
        return torch.cat([mean_p, max_p], dim=-1)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]
            edge_attr: [num_edges] edge weights
            batch: [num_nodes] batch assignment

        Returns:
            logits: [batch_size, num_classes]
        """
        graph_emb = self.encode(x, edge_index, edge_attr, batch)
        return self.classifier(graph_emb)

    @staticmethod
    def from_config(config: dict, in_channels: int) -> "GraphSAGEClassifier":
        return GraphSAGEClassifier(
            in_channels=in_channels,
            hidden_channels=config["gnn"]["hidden_channels"],
            num_classes=config["dataset"]["num_genres"],
            num_layers=config["gnn"]["num_layers"],
            dropout=config["gnn"]["dropout"],
        )


class GATClassifier(nn.Module):
    """
    Enhanced GAT-based multi-label classifier with dual pooling.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 8,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.res_projs.append(
            nn.Linear(in_channels, hidden_channels, bias=False)
            if in_channels != hidden_channels
            else nn.Identity()
        )

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.res_projs.append(nn.Identity())

        pool_dim = 2 * hidden_channels
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

        self.hidden_channels = hidden_channels

    def encode(self, x, edge_index, edge_attr, batch):
        for i, (conv, norm, res) in enumerate(zip(self.convs, self.norms, self.res_projs)):
            identity = res(x)
            h = conv(x, edge_index)
            h = norm(h)
            h = h + identity
            h = F.relu(h)
            x = F.dropout(h, p=self.dropout, training=self.training)

        mean_p = global_mean_pool(x, batch)
        max_p = global_max_pool(x, batch)
        return torch.cat([mean_p, max_p], dim=-1)

    def forward(self, x, edge_index, edge_attr, batch):
        graph_emb = self.encode(x, edge_index, edge_attr, batch)
        return self.classifier(graph_emb)
