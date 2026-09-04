"""
Contrastive learning module for Task 4.

Architecture:
  Audio graph → GNN → projection head → normalised audio embedding
  Caption     → BERT → projection head → normalised text embedding

Loss: InfoNCE (symmetric)

Evaluation: R@1, R@5, R@10 for both caption→audio and audio→caption.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import SAGEConv, global_mean_pool


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ContrastiveGNNBert(nn.Module):
    """
    Contrastive model: learns aligned audio-text embeddings.

    Uses InfoNCE loss to push matching (audio, text) pairs together
    and non-matching pairs apart.
    """

    def __init__(
        self,
        gnn_in_channels: int,
        bert_model_name: str = "distilbert-base-uncased",
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        gnn_dropout: float = 0.3,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        # --- BERT ---
        self.bert = AutoModel.from_pretrained(bert_model_name, attn_implementation="eager")
        bert_hidden = self.bert.config.hidden_size
        # Freeze BERT for contrastive (usually more stable)
        for param in self.bert.parameters():
            param.requires_grad = False

        # --- GNN ---
        self.gnn_convs = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        self.gnn_convs.append(SAGEConv(gnn_in_channels, gnn_hidden))
        self.gnn_norms.append(nn.BatchNorm1d(gnn_hidden))
        for _ in range(gnn_layers - 1):
            self.gnn_convs.append(SAGEConv(gnn_hidden, gnn_hidden))
            self.gnn_norms.append(nn.BatchNorm1d(gnn_hidden))
        self.gnn_dropout = gnn_dropout

        # --- Projection heads ---
        self.audio_proj = ProjectionHead(gnn_hidden, projection_dim)
        self.text_proj = ProjectionHead(bert_hidden, projection_dim)

    def encode_audio(self, x, edge_index, edge_attr, batch):
        h = x
        for i, (conv, norm) in enumerate(zip(self.gnn_convs, self.gnn_norms)):
            h = conv(h, edge_index)
            h = norm(h)
            if i < len(self.gnn_convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.gnn_dropout, training=self.training)
        graph_emb = global_mean_pool(h, batch)  # [B, gnn_hidden]
        return self.audio_proj(graph_emb)

    def encode_text(self, input_ids, attention_mask, batch_size=None):
        if input_ids.dim() == 1 and batch_size is not None:
            input_ids = input_ids.view(batch_size, -1)
        if attention_mask.dim() == 1 and batch_size is not None:
            attention_mask = attention_mask.view(batch_size, -1)
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = bert_out.last_hidden_state[:, 0, :]  # [B, bert_hidden]
        return self.text_proj(cls)

    def forward(self, input_ids, attention_mask, x, edge_index, edge_attr, batch):
        """
        Returns: audio_emb [B, proj_dim], text_emb [B, proj_dim]
        """
        b_size = batch.max().item() + 1 if batch is not None else None
        audio_emb = self.encode_audio(x, edge_index, edge_attr, batch)
        text_emb = self.encode_text(input_ids, attention_mask, batch_size=b_size)
        return audio_emb, text_emb

    def info_nce_loss(self, audio_emb, text_emb):
        """
        Symmetric InfoNCE loss.

        Args:
            audio_emb: [B, D] L2-normalised
            text_emb:  [B, D] L2-normalised

        Returns:
            scalar loss
        """
        # Similarity matrix
        logits = audio_emb @ text_emb.T / self.temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.T, labels)

        return (loss_a2t + loss_t2a) / 2

    @staticmethod
    def from_config(config: dict, gnn_in_channels: int) -> "ContrastiveGNNBert":
        return ContrastiveGNNBert(
            gnn_in_channels=gnn_in_channels,
            bert_model_name=config["bert"]["model_name"],
            gnn_hidden=config["gnn"]["hidden_channels"],
            gnn_layers=config["gnn"]["num_layers"],
            gnn_dropout=config["gnn"]["dropout"],
            projection_dim=config["contrastive"]["projection_dim"],
            temperature=config["contrastive"]["temperature"],
        )


def compute_retrieval_metrics(audio_emb, text_emb, ks=(1, 5, 10)):
    """
    Compute recall@K for audio→text and text→audio retrieval.

    Args:
        audio_emb: [N, D] normalised
        text_emb:  [N, D] normalised
        ks: tuple of K values

    Returns:
        dict with 'a2t_R@K' and 't2a_R@K' entries
    """
    sim = audio_emb @ text_emb.T  # [N, N]
    N = sim.size(0)
    gt = torch.arange(N, device=sim.device)

    # Guard against K > N (e.g. a small validation set or a trailing partial
    # batch): torch.topk would otherwise raise a RuntimeError. Any K beyond
    # the number of candidates is trivially satisfied (recall saturates at 1).
    max_k = max(ks)
    safe_k = min(max_k, N)

    metrics = {}

    # Audio → Text
    _, a2t_ranks = sim.topk(safe_k, dim=1)
    for k in ks:
        k_eff = min(k, N)
        hits = (a2t_ranks[:, :k_eff] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"a2t_R@{k}"] = hits

    # Text → Audio
    _, t2a_ranks = sim.T.topk(safe_k, dim=1)
    for k in ks:
        k_eff = min(k, N)
        hits = (t2a_ranks[:, :k_eff] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"t2a_R@{k}"] = hits

    return metrics
