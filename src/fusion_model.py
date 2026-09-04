"""
GNN-BERT Fusion models for Task 3.

1. CrossAttentionFusion — the primary model
   GNN graph embedding as Query, BERT hidden states as Key/Value
   z = concat(g, cross_attn_context) → classifier

2. EarlyFusion — ablation baseline
   z = concat(GNN_pool, BERT_cls) → classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import SAGEConv, global_mean_pool


# GNN Encoder (shared by fusion models)

class GNNEncoder(nn.Module):
    """Standalone GNN encoder that produces node-level embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.hidden_channels = hidden_channels

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Returns:
            node_emb: [total_nodes, hidden_channels]
            graph_emb: [batch_size, hidden_channels]
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = global_mean_pool(x, batch)
        return x, graph_emb


# Task 3: Cross-Attention Fusion

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: GNN queries attend to BERT key/values.

    Htext = BERT(text)            → [B, seq_len, bert_dim]
    g_nodes, g_pool = GNN(graph)  → nodes: [B, max_nodes, gnn_dim], pool: [B, gnn_dim]

    Q = g_pool @ WQ               → [B, 1, latent_dim]
    K = Htext @ WK                → [B, seq_len, latent_dim]
    V = Htext @ WV                → [B, seq_len, latent_dim]

    A = softmax(QK^T / sqrt(d))   → [B, 1, seq_len]
    context = A @ V                → [B, 1, latent_dim]

    z = concat(g_pool_proj, context)  → [B, 2 * latent_dim]
    prediction = classifier(z)
    """

    def __init__(
        self,
        gnn_in_channels: int,
        num_classes: int = 8,
        bert_model_name: str = "distilbert-base-uncased",
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        gnn_dropout: float = 0.3,
        latent_dim: int = 256,
        num_heads: int = 8,
        fusion_dropout: float = 0.3,
        freeze_mode: str = "partial",
        freeze_layers: int = 4,
    ):
        super().__init__()

        # --- BERT encoder ---
        self.bert = AutoModel.from_pretrained(bert_model_name, attn_implementation="eager")
        bert_hidden = self.bert.config.hidden_size
        self._apply_bert_freeze(freeze_mode, freeze_layers)

        # --- GNN encoder ---
        self.gnn = GNNEncoder(gnn_in_channels, gnn_hidden, gnn_layers, gnn_dropout)

        # --- Projection layers ---
        self.latent_dim = latent_dim
        self.text_proj = nn.Linear(bert_hidden, latent_dim)
        self.audio_proj = nn.Linear(gnn_hidden, latent_dim)

        # --- Cross-attention ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True
        )

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def _apply_bert_freeze(self, mode: str, freeze_layers: int):
        if mode == "frozen":
            for param in self.bert.parameters():
                param.requires_grad = False
        elif mode == "partial":
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            layers = None
            if hasattr(self.bert, "transformer"):
                layers = self.bert.transformer.layer
            elif hasattr(self.bert, "encoder"):
                layers = self.bert.encoder.layer
            if layers is not None:
                for i, layer in enumerate(layers):
                    if i < freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

    def forward(
        self,
        input_ids,       # [B, seq_len]
        attention_mask,  # [B, seq_len]
        x,               # [total_nodes, in_channels]
        edge_index,      # [2, total_edges]
        edge_attr,       # [total_edges]
        batch,           # [total_nodes]
        return_attention: bool = False,
    ):
        if input_ids.dim() == 1:
            b_size = batch.max().item() + 1 if batch is not None else 1
            input_ids = input_ids.view(b_size, -1)
        if attention_mask.dim() == 1:
            b_size = batch.max().item() + 1 if batch is not None else 1
            attention_mask = attention_mask.view(b_size, -1)

        # --- Text encoding ---
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        Htext = bert_out.last_hidden_state  # [B, seq_len, bert_hidden]

        # --- Audio encoding ---
        node_emb, graph_emb = self.gnn(x, edge_index, edge_attr, batch)
        # graph_emb: [B, gnn_hidden]

        # --- Project to latent space ---
        text_proj = self.text_proj(Htext)           # [B, seq_len, latent_dim]
        audio_proj = self.audio_proj(graph_emb)     # [B, latent_dim]
        audio_query = audio_proj.unsqueeze(1)       # [B, 1, latent_dim]

        # --- Cross-attention: GNN queries BERT ---
        # key_padding_mask: True where padding tokens should be ignored
        key_padding_mask = (attention_mask == 0)  # [B, seq_len]

        attn_output, attn_weights = self.cross_attn(
            query=audio_query,       # [B, 1, latent_dim]
            key=text_proj,           # [B, seq_len, latent_dim]
            value=text_proj,         # [B, seq_len, latent_dim]
            key_padding_mask=key_padding_mask,
        )
        # attn_output: [B, 1, latent_dim]
        context = attn_output.squeeze(1)  # [B, latent_dim]

        # --- Fusion ---
        z = torch.cat([audio_proj, context], dim=1)  # [B, 2*latent_dim]

        # --- Classification ---
        logits = self.classifier(z)

        if return_attention:
            return logits, attn_weights  # attn_weights: [B, 1, seq_len]
        return logits

    @staticmethod
    def from_config(config: dict, gnn_in_channels: int) -> "CrossAttentionFusion":
        return CrossAttentionFusion(
            gnn_in_channels=gnn_in_channels,
            num_classes=config["dataset"]["num_genres"],
            bert_model_name=config["bert"]["model_name"],
            gnn_hidden=config["gnn"]["hidden_channels"],
            gnn_layers=config["gnn"]["num_layers"],
            gnn_dropout=config["gnn"]["dropout"],
            latent_dim=config["fusion"]["latent_dim"],
            num_heads=config["fusion"]["num_heads"],
            fusion_dropout=config["fusion"]["dropout"],
            freeze_mode=config["bert"]["freeze_mode"],
            freeze_layers=config["bert"]["freeze_layers"],
        )


# Ablation: Early Concatenation Fusion

class EarlyFusion(nn.Module):
    """
    Simple early fusion: concat(BERT_CLS, GNN_pool) → MLP → multi-label.
    Used as an ablation baseline against cross-attention.
    """

    def __init__(
        self,
        gnn_in_channels: int,
        num_classes: int = 8,
        bert_model_name: str = "distilbert-base-uncased",
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        gnn_dropout: float = 0.3,
        fusion_dropout: float = 0.3,
        freeze_mode: str = "partial",
        freeze_layers: int = 4,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name, attn_implementation="eager")
        bert_hidden = self.bert.config.hidden_size
        self._apply_bert_freeze(freeze_mode, freeze_layers)

        self.gnn = GNNEncoder(gnn_in_channels, gnn_hidden, gnn_layers, gnn_dropout)

        concat_dim = bert_hidden + gnn_hidden
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(256, num_classes),
        )

    def _apply_bert_freeze(self, mode, freeze_layers):
        if mode == "frozen":
            for p in self.bert.parameters():
                p.requires_grad = False
        elif mode == "partial":
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False
            layers = getattr(self.bert, "transformer", None)
            if layers:
                layers = layers.layer
            else:
                layers = getattr(self.bert.encoder, "layer", None)
            if layers:
                for i, l in enumerate(layers):
                    if i < freeze_layers:
                        for p in l.parameters():
                            p.requires_grad = False

    def forward(self, input_ids, attention_mask, x, edge_index, edge_attr, batch,
                return_attention=False):
        if input_ids.dim() == 1:
            b_size = batch.max().item() + 1 if batch is not None else 1
            input_ids = input_ids.view(b_size, -1)
        if attention_mask.dim() == 1:
            b_size = batch.max().item() + 1 if batch is not None else 1
            attention_mask = attention_mask.view(b_size, -1)

        # BERT CLS
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = bert_out.last_hidden_state[:, 0, :]  # [B, bert_hidden]

        # GNN pool
        _, graph_emb = self.gnn(x, edge_index, edge_attr, batch)  # [B, gnn_hidden]

        # Concat + classify
        z = torch.cat([cls, graph_emb], dim=1)
        logits = self.classifier(z)

        if return_attention:
            return logits, None
        return logits

    @staticmethod
    def from_config(config: dict, gnn_in_channels: int) -> "EarlyFusion":
        return EarlyFusion(
            gnn_in_channels=gnn_in_channels,
            num_classes=config["dataset"]["num_genres"],
            bert_model_name=config["bert"]["model_name"],
            gnn_hidden=config["gnn"]["hidden_channels"],
            gnn_layers=config["gnn"]["num_layers"],
            gnn_dropout=config["gnn"]["dropout"],
            fusion_dropout=config["fusion"]["dropout"],
            freeze_mode=config["bert"]["freeze_mode"],
            freeze_layers=config["bert"]["freeze_layers"],
        )
