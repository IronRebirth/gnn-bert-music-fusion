"""
Baselines: CNN mel-spectrogram classifier and random/majority baselines.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class CNNBaseline(nn.Module):
    """
    CNN baseline for audio-only classification on log-mel spectrograms.

    Architecture:
      [1, n_mels, T] → Conv blocks → AdaptiveAvgPool → FC → multi-label

    Kept lightweight for Kaggle.
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = 8,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]

        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 1, n_mels, T]
        Returns:
            logits: [batch, num_classes]
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    @staticmethod
    def from_config(config: dict) -> "CNNBaseline":
        return CNNBaseline(
            n_mels=config["dataset"]["n_mels"],
            num_classes=config["dataset"]["num_genres"],
            channels=config["cnn"]["channels"],
            kernel_size=config["cnn"]["kernel_size"],
            dropout=config["cnn"]["dropout"],
        )


class AudioMLPBaseline(nn.Module):
    """
    MLP baseline operating directly on pooled audio segment features (mel_mean + chroma_mean).
    Allows direct evaluation of performance gained by graph structure vs flat MLP.
    """

    def __init__(self, in_features: int = 140, hidden_dim: int = 128, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class RandomBaseline:
    """Random predictions baseline for comparison."""

    def __init__(self, num_classes: int, seed: int = 42):
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)

    def predict(self, n_samples: int) -> np.ndarray:
        return self.rng.randint(0, 2, size=(n_samples, self.num_classes)).astype(np.float32)


class MajorityBaseline:
    """Predict the most frequent label(s) or prior distribution from training set."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.priors = np.zeros(num_classes, dtype=np.float32)
        self.majority = np.zeros(num_classes, dtype=np.float32)

    def fit(self, labels: np.ndarray, threshold: float = 0.5):
        """labels: [n_samples, num_classes] multi-hot."""
        self.priors = labels.mean(axis=0).astype(np.float32)
        self.majority = (self.priors >= threshold).astype(np.float32)

    def predict(self, n_samples: int) -> np.ndarray:
        return np.tile(self.majority, (n_samples, 1))

    def predict_proba(self, n_samples: int) -> np.ndarray:
        return np.tile(self.priors, (n_samples, 1))
