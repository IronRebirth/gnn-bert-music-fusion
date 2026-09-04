"""
Improved CNN and CNN-BERT Fusion models.

1. ImprovedCNN — Enhanced CNN with residual blocks, dual pooling, multi-segment support
2. FocalLoss — Focal loss for class imbalance
3. CNNBertFusion — CNN audio + BERT text fusion with gated/concat/cross-attention
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# Focal Loss for class imbalance

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces loss for well-classified examples, focusing on hard examples.
    Particularly useful for imbalanced multi-label datasets.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw logits
            targets: [B, C] multi-hot labels

        Returns:
            scalar loss
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * focal_weight * bce

        return loss.mean()


# Residual Conv Block

class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block: Conv2d → BN → Act → Conv2d → BN + Skip → Act → Pool.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        activation: str = "gelu",
        pool: bool = True,
    ):
        super().__init__()
        pad = kernel_size // 2

        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = act_fn

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.GELU() if activation == "gelu" else nn.ReLU()

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + identity)
        out = self.pool(out)
        return out


# Improved CNN

class ImprovedCNN(nn.Module):
    """
    Enhanced CNN for audio classification on log-mel spectrograms.

    Improvements over CNNBaseline:
      - Residual convolutional blocks
      - Dual pooling (global avg + global max)
      - Stronger MLP head with LayerNorm
      - GELU activation
      - Multi-segment support (process all segments, pool embeddings)

    Architecture:
      [B, 1, n_mels, T] → ResConvBlock×N → DualPool → MLP → logits

    For multi-segment mode:
      [B, S, 1, n_mels, T] → CNN per segment → mean pool → MLP → logits
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = 8,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.4,
        activation: str = "gelu",
        use_dual_pool: bool = True,
        head_hidden: int = 256,
    ):
        super().__init__()

        if channels is None:
            channels = [64, 128, 256]

        # Build residual conv blocks
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(ResidualConvBlock(in_ch, out_ch, kernel_size, activation))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        # Pooling
        self.use_dual_pool = use_dual_pool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        pool_dim = channels[-1] * 2 if use_dual_pool else channels[-1]

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pool_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden // 2, num_classes),
        )

        self.embed_dim = pool_dim  # Exposed for fusion models

    def _encode_single(self, x):
        """
        Encode a single spectrogram.
        Args: x [B, 1, n_mels, T]
        Returns: embedding [B, pool_dim]
        """
        h = self.features(x)
        if self.use_dual_pool:
            avg = self.avg_pool(h).flatten(1)
            mx = self.max_pool(h).flatten(1)
            return torch.cat([avg, mx], dim=-1)
        else:
            return self.avg_pool(h).flatten(1)

    def encode(self, x, seg_mask=None):
        """
        Encode spectrograms, supporting multi-segment input.

        Args:
            x: [B, 1, n_mels, T] single segment
               OR [B, S, 1, n_mels, T] multi-segment
            seg_mask: [B, S] bool mask for valid segments (multi-segment only)

        Returns:
            embedding: [B, pool_dim]
        """
        if x.dim() == 5:
            # Multi-segment: [B, S, 1, n_mels, T]
            B, S, C, H, W = x.shape
            x_flat = x.view(B * S, C, H, W)  # [B*S, 1, n_mels, T]
            emb_flat = self._encode_single(x_flat)  # [B*S, pool_dim]
            emb = emb_flat.view(B, S, -1)  # [B, S, pool_dim]

            if seg_mask is not None:
                # Masked mean pooling
                mask = seg_mask.unsqueeze(-1).float()  # [B, S, 1]
                emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                emb = emb.mean(dim=1)

            return emb  # [B, pool_dim]
        else:
            return self._encode_single(x)

    def forward(self, x, seg_mask=None):
        """
        Args:
            x: [B, 1, n_mels, T] or [B, S, 1, n_mels, T]
            seg_mask: [B, S] optional segment mask

        Returns:
            logits: [B, num_classes]
        """
        emb = self.encode(x, seg_mask)
        return self.classifier(emb)

    @staticmethod
    def from_config(config: dict) -> "ImprovedCNN":
        cfg = config.get("improved_cnn", {})
        return ImprovedCNN(
            n_mels=config["dataset"]["n_mels"],
            num_classes=config["dataset"]["num_genres"],
            channels=cfg.get("channels", [64, 128, 256]),
            kernel_size=cfg.get("kernel_size", 3),
            dropout=cfg.get("dropout", 0.4),
            activation=cfg.get("activation", "gelu"),
            use_dual_pool=cfg.get("use_dual_pool", True),
            head_hidden=cfg.get("head_hidden", 256),
        )


# CNN-BERT Gated Fusion

class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining two embedding modalities.

    gate = sigmoid(W_a * audio + W_t * text + b)
    fused = gate * audio_proj + (1 - gate) * text_proj
    """

    def __init__(self, audio_dim: int, text_dim: int, latent_dim: int):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, latent_dim)
        self.text_proj = nn.Linear(text_dim, latent_dim)
        self.gate_audio = nn.Linear(audio_dim, latent_dim)
        self.gate_text = nn.Linear(text_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, audio_emb, text_emb):
        """
        Args:
            audio_emb: [B, audio_dim]
            text_emb: [B, text_dim]

        Returns:
            fused: [B, latent_dim]
        """
        gate = torch.sigmoid(self.gate_audio(audio_emb) + self.gate_text(text_emb))
        a_proj = self.audio_proj(audio_emb)
        t_proj = self.text_proj(text_emb)
        fused = gate * a_proj + (1 - gate) * t_proj
        return self.norm(fused)


# CNN-BERT Fusion Model

class CNNBertFusion(nn.Module):
    """
    CNN + BERT fusion for multi-label music classification.

    Architecture:
      Mel Spectrogram → ImprovedCNN → Audio Embedding
      Text → BERT → [CLS] → Text Embedding
      → Gated Fusion / Concat / Cross-Attention
      → Residual MLP → Multi-label logits

    This replaces GNN with CNN as the audio backbone,
    using the spectrogram directly rather than time-averaged graph features.
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = 8,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.4,
        cnn_activation: str = "gelu",
        cnn_head_hidden: int = 256,
        bert_model_name: str = "distilbert-base-uncased",
        fusion_type: str = "gated",
        latent_dim: int = 256,
        fusion_dropout: float = 0.3,
        freeze_mode: str = "partial",
        freeze_layers: int = 4,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        # --- CNN audio encoder ---
        self.cnn = ImprovedCNN(
            n_mels=n_mels,
            num_classes=num_classes,  # Not used directly, we use encode()
            channels=cnn_channels or [64, 128, 256],
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
            activation=cnn_activation,
            use_dual_pool=True,
            head_hidden=cnn_head_hidden,
        )
        cnn_embed_dim = self.cnn.embed_dim  # e.g., 512 with dual pool

        # --- BERT text encoder ---
        self.bert = AutoModel.from_pretrained(bert_model_name, attn_implementation="eager")
        bert_hidden = self.bert.config.hidden_size
        self.bert_dropout = nn.Dropout(0.1)
        self._apply_bert_freeze(freeze_mode, freeze_layers)

        # --- Fusion ---
        if fusion_type == "gated":
            self.fusion = GatedFusion(cnn_embed_dim, bert_hidden, latent_dim)
            classifier_input_dim = latent_dim
        elif fusion_type == "concat":
            self.audio_proj = nn.Linear(cnn_embed_dim, latent_dim)
            self.text_proj = nn.Linear(bert_hidden, latent_dim)
            classifier_input_dim = latent_dim * 2
        elif fusion_type == "cross_attention":
            self.audio_proj = nn.Linear(cnn_embed_dim, latent_dim)
            self.text_proj = nn.Linear(bert_hidden, latent_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=latent_dim, num_heads=8, batch_first=True,
            )
            classifier_input_dim = latent_dim * 2
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(fusion_dropout * 0.5),
            nn.Linear(latent_dim // 2, num_classes),
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

    def get_parameter_groups(self, bert_lr: float = 2e-5, new_lr: float = 3e-4):
        """
        Return parameter groups with differential learning rates.
        BERT fine-tunable params get bert_lr, everything else gets new_lr.
        """
        bert_params = []
        new_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("bert."):
                bert_params.append(param)
            else:
                new_params.append(param)

        return [
            {"params": bert_params, "lr": bert_lr},
            {"params": new_params, "lr": new_lr},
        ]

    def forward(self, mel, input_ids, attention_mask, seg_mask=None):
        """
        Args:
            mel: [B, 1, n_mels, T] or [B, S, 1, n_mels, T]
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            seg_mask: [B, S] optional segment mask

        Returns:
            logits: [B, num_classes]
        """
        # Audio encoding
        audio_emb = self.cnn.encode(mel, seg_mask)  # [B, cnn_embed_dim]

        # Text encoding
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self.bert_dropout(bert_out.last_hidden_state[:, 0, :])  # [B, bert_hidden]

        # Fusion
        if self.fusion_type == "gated":
            fused = self.fusion(audio_emb, text_emb)  # [B, latent_dim]
        elif self.fusion_type == "concat":
            a_proj = self.audio_proj(audio_emb)
            t_proj = self.text_proj(text_emb)
            fused = torch.cat([a_proj, t_proj], dim=-1)  # [B, 2*latent_dim]
        elif self.fusion_type == "cross_attention":
            a_proj = self.audio_proj(audio_emb).unsqueeze(1)  # [B, 1, latent_dim]
            # Use full BERT hidden states as K/V
            text_hidden = self.text_proj(bert_out.last_hidden_state)  # [B, seq_len, latent_dim]
            key_padding_mask = (attention_mask == 0)
            attn_out, _ = self.cross_attn(
                query=a_proj, key=text_hidden, value=text_hidden,
                key_padding_mask=key_padding_mask,
            )
            context = attn_out.squeeze(1)  # [B, latent_dim]
            fused = torch.cat([a_proj.squeeze(1), context], dim=-1)  # [B, 2*latent_dim]

        return self.classifier(fused)

    @staticmethod
    def from_config(config: dict) -> "CNNBertFusion":
        cfg = config.get("cnn_bert_fusion", {})
        icfg = config.get("improved_cnn", {})
        return CNNBertFusion(
            n_mels=config["dataset"]["n_mels"],
            num_classes=config["dataset"]["num_genres"],
            cnn_channels=icfg.get("channels", [64, 128, 256]),
            cnn_kernel_size=icfg.get("kernel_size", 3),
            cnn_dropout=icfg.get("dropout", 0.4),
            cnn_activation=icfg.get("activation", "gelu"),
            cnn_head_hidden=icfg.get("head_hidden", 256),
            bert_model_name=config["bert"]["model_name"],
            fusion_type=cfg.get("fusion_type", "gated"),
            latent_dim=cfg.get("latent_dim", 256),
            fusion_dropout=cfg.get("dropout", 0.3),
            freeze_mode=config["bert"]["freeze_mode"],
            freeze_layers=config["bert"]["freeze_layers"],
        )
