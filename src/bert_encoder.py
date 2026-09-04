"""
BERT-based text classifier for Task 1.

Architecture:
  Text → BERT → [CLS] → Dropout → Linear → Sigmoid → Multi-label predictions

Supports frozen / partially fine-tuned / fully fine-tuned BERT.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class BertClassifier(nn.Module):
    """
    BERT-based multi-label classifier.

    Args:
        model_name: HuggingFace model name (e.g. 'distilbert-base-uncased')
        num_classes: number of output labels
        dropout: dropout rate before classifier
        freeze_mode: 'frozen', 'partial', or 'full'
        freeze_layers: number of transformer layers to freeze (partial mode)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 8,
        dropout: float = 0.3,
        freeze_mode: str = "partial",
        freeze_layers: int = 4,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        hidden_size = self.bert.config.hidden_size  # 768 for distilbert

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Apply freeze strategy
        self._apply_freeze(freeze_mode, freeze_layers)

    def _apply_freeze(self, mode: str, freeze_layers: int):
        if mode == "frozen":
            for param in self.bert.parameters():
                param.requires_grad = False
        elif mode == "partial":
            # Freeze embeddings
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # Freeze first N transformer layers
            # distilbert uses .transformer.layer, bert uses .encoder.layer
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
        # mode == "full" → everything trainable (default)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, num_classes]  (raw, apply sigmoid externally)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

    @staticmethod
    def from_config(config: dict) -> "BertClassifier":
        """Create BertClassifier from config dict."""
        return BertClassifier(
            model_name=config["bert"]["model_name"],
            num_classes=config["dataset"]["num_genres"],
            # Was hard-wired to config["gnn"]["dropout"] — an unrelated
            # section that happens to share the same default value. Use a
            # BERT-specific key (falls back to 0.3 if not set) so tuning the
            # GNN's dropout no longer silently changes the text model too.
            dropout=config["bert"].get("dropout", 0.3),
            freeze_mode=config["bert"]["freeze_mode"],
            freeze_layers=config["bert"]["freeze_layers"],
        )
