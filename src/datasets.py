"""
Dataset classes for all tasks.

- FMAMetadata: loads FMA tracks.csv, produces track_id → (genre_labels, text)
- BertDataset: tokenized text + multi-label for Task 1
- GraphDataset: PyG graph + multi-label for Task 2
- MelDataset: mel spectrograms + multi-label for CNN baseline
- FusionDataset: text + graph + multi-label for Task 3
- ContrastiveDataset: text + graph pairs for Task 4
"""
from __future__ import annotations


import ast
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import AutoTokenizer

from src.utils import get_genre_labels


# FMA Metadata Loader

class FMAMetadata:
    """
    Parse FMA tracks.csv and produce usable mappings.

    Provides:
      - track_id → multi-hot genre label
      - track_id → text string (genre names + tags + title)
      - artist_id mapping for leak-free splits
      - track lists per split (train/val/test)
    """

    def __init__(self, metadata_dir: str, subset: str = "small"):
        self.metadata_dir = metadata_dir
        self.subset = subset
        self.genre_labels = get_genre_labels()  # 8 sorted genres
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_labels)}
        self.num_classes = len(self.genre_labels)

        self._load()

    def _load(self):
        """Parse the 3-row-header FMA tracks.csv into a clean dataframe."""
        tracks_path = os.path.join(self.metadata_dir, "tracks.csv")
        raw = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

        # Filter to subset
        mask = raw[("set", "subset")] == self.subset
        df = raw.loc[mask].copy()

        self.track_ids = sorted(df.index.tolist())

        # Genre labels (top-level)
        self._genre_top = df[("track", "genre_top")].to_dict()

        # All genres (list of genre IDs) — for multi-label
        self._genres_all = df[("track", "genres_all")].to_dict()

        # Tags
        self._tags = df[("track", "tags")].to_dict()

        # Title
        self._title = df[("track", "title")].to_dict()

        # Artist info
        self._artist_id = df[("artist", "id")].to_dict()
        self._artist_name = df[("artist", "name")].to_dict()

        # FMA predefined splits
        self._split = df[("set", "split")].to_dict()

        # Load genre_id → name mapping
        genres_path = os.path.join(self.metadata_dir, "genres.csv")
        genres_df = pd.read_csv(genres_path)
        self._genre_id_to_name = dict(zip(genres_df["genre_id"], genres_df["title"]))

    def get_multi_hot_label(self, track_id: int) -> torch.Tensor:
        """
        Return multi-hot tensor of shape [num_classes] for the track.
        Uses genres_all to find all applicable top-level genres.
        """
        label = torch.zeros(self.num_classes, dtype=torch.float)
        # Primary genre
        genre_top = self._genre_top.get(track_id, "")
        if isinstance(genre_top, str) and genre_top in self.genre_to_idx:
            label[self.genre_to_idx[genre_top]] = 1.0

        # Also check genres_all for multi-label
        genres_all_str = self._genres_all.get(track_id, "[]")
        try:
            genre_ids = ast.literal_eval(str(genres_all_str))
            if isinstance(genre_ids, list):
                for gid in genre_ids:
                    gname = self._genre_id_to_name.get(gid, "")
                    if gname in self.genre_to_idx:
                        label[self.genre_to_idx[gname]] = 1.0
        except (ValueError, SyntaxError):
            pass

        # Safety: ensure at least the top genre is set
        if label.sum() == 0 and isinstance(genre_top, str) and genre_top in self.genre_to_idx:
            label[self.genre_to_idx[genre_top]] = 1.0

        return label

    def get_text(self, track_id: int) -> str:
        """
        Construct a text description for the track from available metadata (Title, Artist, Tags)
        WITHOUT including the ground-truth target genre to prevent label leakage.
        """
        parts = []

        title = self._title.get(track_id, "")
        if isinstance(title, str) and title:
            parts.append(f"Title: {title}")

        artist = self._artist_name.get(track_id, "")
        if isinstance(artist, str) and artist:
            parts.append(f"Artist: {artist}")

        tags = self._tags.get(track_id, "")
        if isinstance(tags, str) and tags and tags != "[]":
            try:
                tag_list = ast.literal_eval(tags)
                if isinstance(tag_list, list) and tag_list:
                    parts.append(f"Tags: {', '.join(str(t) for t in tag_list)}")
            except (ValueError, SyntaxError):
                if tags.strip():
                    parts.append(f"Tags: {tags}")

        return ". ".join(parts) if parts else "Music track"

    def get_artist_id(self, track_id: int):
        return self._artist_id.get(track_id, None)

    def get_split(self, track_id: int) -> str:
        """Return the FMA predefined split: 'training', 'validation', 'test'."""
        return str(self._split.get(track_id, "training"))

    def get_split_track_ids(self) -> dict:
        """
        Return dict with keys 'train', 'val', 'test' → list of track_ids.
        Uses the FMA predefined split column, preventing artist leakage.
        """
        splits = {"train": [], "val": [], "test": []}
        for tid in self.track_ids:
            s = self.get_split(tid)
            if s == "training":
                splits["train"].append(tid)
            elif s == "validation":
                splits["val"].append(tid)
            elif s == "test":
                splits["test"].append(tid)
            else:
                splits["train"].append(tid)  # fallback
        return splits


# Task 1: BERT Text Dataset

class BertDataset(Dataset):
    """
    Dataset for BERT-only classification (Task 1).
    Each sample: (input_ids, attention_mask, multi-hot label).
    """

    def __init__(
        self,
        track_ids: list[int],
        metadata: FMAMetadata,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        cache_dir: str | None = None,
    ):
        self.track_ids = track_ids
        self.metadata = metadata
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.cache_dir = cache_dir

        # Pre-tokenize if cache doesn't exist
        self._cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            h = hash(tuple(sorted(track_ids))) % (10**10)
            self._cache_path = os.path.join(cache_dir, f"bert_cache_{h}.pt")

        self._data = self._load_or_build()

    def _load_or_build(self):
        if self._cache_path and os.path.exists(self._cache_path):
            return torch.load(self._cache_path, weights_only=False)

        data = []
        for tid in self.track_ids:
            text = self.metadata.get_text(tid)
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            label = self.metadata.get_multi_hot_label(tid)
            data.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": label,
                "track_id": tid,
            })

        if self._cache_path:
            torch.save(data, self._cache_path)

        return data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


# Task 2: Graph Dataset

class GraphDataset(Dataset):
    """
    Dataset for GNN classification (Task 2).
    Each sample: PyG Data object with node features, edges, and label.
    """

    def __init__(
        self,
        track_ids: list[int],
        metadata: FMAMetadata,
        graphs_dir: str,
    ):
        self.metadata = metadata
        self.graphs_dir = graphs_dir

        # Filter to tracks that have graph files
        self.track_ids = []
        for tid in track_ids:
            graph_path = os.path.join(graphs_dir, f"{tid:06d}_graph.pt")
            if os.path.exists(graph_path):
                self.track_ids.append(tid)

        if len(self.track_ids) < len(track_ids):
            print(
                f"  GraphDataset: {len(self.track_ids)}/{len(track_ids)} "
                f"tracks have graph files"
            )

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        tid = self.track_ids[idx]
        graph_path = os.path.join(self.graphs_dir, f"{tid:06d}_graph.pt")
        graph = torch.load(graph_path, weights_only=False)

        label = self.metadata.get_multi_hot_label(tid)
        graph.y = label.unsqueeze(0)
        graph.track_id = tid

        return graph


# CNN Baseline: Mel Spectrogram Dataset

class MelDataset(Dataset):
    """
    Dataset for CNN baseline: loads first segment's mel spectrogram.
    Each sample: (mel_tensor [1, n_mels, T], label).
    """

    def __init__(
        self,
        track_ids: list[int],
        metadata: FMAMetadata,
        features_dir: str,
    ):
        self.metadata = metadata
        self.features_dir = features_dir

        self.track_ids = []
        for tid in track_ids:
            feat_path = os.path.join(features_dir, f"{tid:06d}_features.pt")
            if os.path.exists(feat_path):
                self.track_ids.append(tid)

        if len(self.track_ids) < len(track_ids):
            print(
                f"  MelDataset: {len(self.track_ids)}/{len(track_ids)} "
                f"tracks have feature files"
            )

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        tid = self.track_ids[idx]
        feat_path = os.path.join(self.features_dir, f"{tid:06d}_features.pt")
        segments = torch.load(feat_path, weights_only=False)

        # Use first segment's log-mel spectrogram
        mel = segments[0]["log_mel"]  # [n_mels, T]
        mel_tensor = torch.tensor(mel, dtype=torch.float).unsqueeze(0)  # [1, n_mels, T]

        label = self.metadata.get_multi_hot_label(tid)
        return mel_tensor, label


# Task 3: Fusion Dataset

class FusionDataset(Dataset):
    """
    Dataset for GNN-BERT fusion (Task 3).
    Each sample: (graph, input_ids, attention_mask, label).

    The graph Data object gets input_ids and attention_mask attached
    so PyG DataLoader can batch everything together.
    """

    def __init__(
        self,
        track_ids: list[int],
        metadata: FMAMetadata,
        graphs_dir: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        cache_dir: str | None = None,
    ):
        self.metadata = metadata
        self.graphs_dir = graphs_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Filter to tracks with graph files
        self.track_ids = []
        for tid in track_ids:
            graph_path = os.path.join(graphs_dir, f"{tid:06d}_graph.pt")
            if os.path.exists(graph_path):
                self.track_ids.append(tid)

        if len(self.track_ids) < len(track_ids):
            print(
                f"  FusionDataset: {len(self.track_ids)}/{len(track_ids)} "
                f"tracks have graph files"
            )

        # Pre-tokenize text
        self._text_cache = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        tid = self.track_ids[idx]

        # Load graph
        graph_path = os.path.join(self.graphs_dir, f"{tid:06d}_graph.pt")
        graph = torch.load(graph_path, weights_only=False)

        # Get text
        text = self.metadata.get_text(tid)
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Attach text tensors to graph object so PyG DataLoader batches them
        graph.input_ids = enc["input_ids"]          # [1, max_length]
        graph.attention_mask = enc["attention_mask"]  # [1, max_length]
        graph.y = self.metadata.get_multi_hot_label(tid).unsqueeze(0)
        graph.track_id = tid

        return graph


# Task 4: Contrastive Dataset

class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning (Task 4).
    Returns (graph, text_input_ids, text_attention_mask).
    No classification labels — uses InfoNCE on paired embeddings.
    """

    def __init__(
        self,
        track_ids: list[int],
        metadata: FMAMetadata,
        graphs_dir: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
    ):
        self.metadata = metadata
        self.graphs_dir = graphs_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.track_ids = [
            tid for tid in track_ids
            if os.path.exists(os.path.join(graphs_dir, f"{tid:06d}_graph.pt"))
        ]

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        tid = self.track_ids[idx]

        graph_path = os.path.join(self.graphs_dir, f"{tid:06d}_graph.pt")
        graph = torch.load(graph_path, weights_only=False)

        text = self.metadata.get_text(tid)
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        graph.input_ids = enc["input_ids"]          # [1, max_length]
        graph.attention_mask = enc["attention_mask"]  # [1, max_length]
        graph.track_id = tid

        return graph


# Improved Mel Dataset: Multi-segment + SpecAugment

class ImprovedMelDataset(Dataset):
    """
    An enhanced dataset for CNN classification that utilizes multiple audio segments
    instead of just the first one. It supports SpecAugment for regularization and
    can randomly sample segments to provide data augmentation during training.
    """

    def __init__(
        self,
        track_ids: list[int],
        metadata: FMAMetadata,
        features_dir: str,
        is_training: bool = False,
        augmentation_config: dict | None = None,
        multi_segment: bool = True,
        random_segment: bool = False,
    ):
        self.metadata = metadata
        self.features_dir = features_dir
        self.is_training = is_training
        self.multi_segment = multi_segment
        self.random_segment = random_segment and is_training

        # Configuration for SpecAugment, if provided
        self.aug_enabled = is_training and augmentation_config is not None and augmentation_config.get("enabled", False)
        if self.aug_enabled:
            self.time_mask_param = augmentation_config.get("time_mask_param", 20)
            self.freq_mask_param = augmentation_config.get("freq_mask_param", 10)
            self.num_time_masks = augmentation_config.get("num_time_masks", 2)
            self.num_freq_masks = augmentation_config.get("num_freq_masks", 2)

        # Build track list by checking for existing feature files
        self.track_ids = []
        for tid in track_ids:
            feat_path = os.path.join(features_dir, f"{tid:06d}_features.pt")
            if os.path.exists(feat_path):
                self.track_ids.append(tid)

        if len(self.track_ids) < len(track_ids):
            print(
                f"  ImprovedMelDataset: {len(self.track_ids)}/{len(track_ids)} "
                f"tracks have feature files"
            )

    def _spec_augment(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to a mel-spectrogram.

        This performs frequency masking followed by time masking.
        """
        # mel shape: [batch, n_mels, T]
        _, n_mels, T = mel.shape
        # Frequency masking
        # NOTE: previously this only applied a single frequency mask
        # regardless of `num_freq_masks`, so SpecAugment was effectively
        # half-strength (2x fewer freq masks than configured) and could
        # occasionally mask past the mel-bin range. Loop it like the time
        # masks below, and clamp the mask width to n_mels like
        # CNNBertDataset._spec_augment does.
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, min(self.freq_mask_param, n_mels), (1,)).item()
            f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
            mel[:, f0:f0 + f, :] = 0.0

        # Time masking
        for _ in range(self.num_time_masks):
            t = torch.randint(0, min(self.time_mask_param, T), (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            mel[:, :, t0:t0 + t] = 0.0

        return mel

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        tid = self.track_ids[idx]
        feat_path = os.path.join(self.features_dir, f"{tid:06d}_features.pt")
        segments = torch.load(feat_path, weights_only=False)
        label = self.metadata.get_multi_hot_label(tid)

        if self.random_segment and self.is_training:
            # Random segment for training augmentation
            seg_idx = torch.randint(0, len(segments), (1,)).item()
            mel = segments[seg_idx]["log_mel"]
            mel_tensor = torch.tensor(mel, dtype=torch.float).unsqueeze(0)  # [1, n_mels, T]
            if self.aug_enabled:
                mel_tensor = self._spec_augment(mel_tensor)
            return mel_tensor, label

        if self.multi_segment:
            # Stack all segments
            mel_list = []
            for seg in segments:
                mel = torch.tensor(seg["log_mel"], dtype=torch.float).unsqueeze(0)  # [1, n_mels, T]
                if self.aug_enabled:
                    mel = self._spec_augment(mel)
                mel_list.append(mel)
            # Return stacked: [num_segments, 1, n_mels, T]
            mel_stack = torch.stack(mel_list, dim=0)
            return mel_stack, label, len(segments)
        else:
            # Single first segment (same as original MelDataset)
            mel = segments[0]["log_mel"]
            mel_tensor = torch.tensor(mel, dtype=torch.float).unsqueeze(0)
            if self.aug_enabled:
                mel_tensor = self._spec_augment(mel_tensor)
            return mel_tensor, label


def improved_mel_collate_fn(batch):
    """
    Custom collate for ImprovedMelDataset with multi-segment mode.

    Each item is either:
      - (mel_stack [S, 1, n_mels, T], label, num_segments)  — multi-segment
      - (mel_tensor [1, n_mels, T], label)                   — single segment

    For multi-segment: pad to max segments, return mask.
    For single segment: standard batching.
    """
    if len(batch[0]) == 3:
        # Multi-segment mode
        mel_stacks, labels, seg_counts = zip(*batch)
        max_segs = max(seg_counts)
        B = len(batch)

        # Determine mel shape from first sample
        _, C, H, W = mel_stacks[0].shape

        # Pad all to max_segs
        padded = torch.zeros(B, max_segs, C, H, W)
        seg_mask = torch.zeros(B, max_segs, dtype=torch.bool)

        for i, (ms, sc) in enumerate(zip(mel_stacks, seg_counts)):
            padded[i, :sc] = ms
            seg_mask[i, :sc] = True

        labels = torch.stack(labels)
        return padded, labels, seg_mask
    else:
        # Single segment mode
        mels, labels = zip(*batch)
        return torch.stack(mels), torch.stack(labels)


# CNN-BERT Fusion Dataset

class CNNBertDataset(Dataset):
    """
    Dataset for CNN-BERT fusion: provides mel spectrogram + tokenized text.

    Each sample: (mel_tensor(s), input_ids, attention_mask, label)
    """

    def __init__(
        self,
        track_ids: list[int],
        metadata: FMAMetadata,
        features_dir: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        is_training: bool = False,
        augmentation_config: dict | None = None,
        multi_segment: bool = True,
    ):
        self.metadata = metadata
        self.features_dir = features_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.is_training = is_training
        self.multi_segment = multi_segment

        # SpecAugment
        self.aug_enabled = is_training and augmentation_config is not None and augmentation_config.get("enabled", False)
        if self.aug_enabled:
            self.time_mask_param = augmentation_config.get("time_mask_param", 20)
            self.freq_mask_param = augmentation_config.get("freq_mask_param", 10)
            self.num_time_masks = augmentation_config.get("num_time_masks", 2)
            self.num_freq_masks = augmentation_config.get("num_freq_masks", 2)

        self.track_ids = []
        for tid in track_ids:
            feat_path = os.path.join(features_dir, f"{tid:06d}_features.pt")
            if os.path.exists(feat_path):
                self.track_ids.append(tid)

        if len(self.track_ids) < len(track_ids):
            print(
                f"  CNNBertDataset: {len(self.track_ids)}/{len(track_ids)} "
                f"tracks have feature files"
            )

    def _spec_augment(self, mel: torch.Tensor) -> torch.Tensor:
        _, n_mels, T = mel.shape
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, min(self.freq_mask_param, n_mels), (1,)).item()
            f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
            mel[:, f0:f0 + f, :] = 0.0
        for _ in range(self.num_time_masks):
            t = torch.randint(0, min(self.time_mask_param, T), (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            mel[:, :, t0:t0 + t] = 0.0
        return mel

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        tid = self.track_ids[idx]

        # Load audio features
        feat_path = os.path.join(self.features_dir, f"{tid:06d}_features.pt")
        segments = torch.load(feat_path, weights_only=False)

        # Process mel spectrograms
        if self.multi_segment:
            mel_list = []
            for seg in segments:
                mel = torch.tensor(seg["log_mel"], dtype=torch.float).unsqueeze(0)
                if self.aug_enabled:
                    mel = self._spec_augment(mel)
                mel_list.append(mel)
            mel_stack = torch.stack(mel_list, dim=0)  # [S, 1, n_mels, T]
            num_segs = len(segments)
        else:
            if self.is_training and self.aug_enabled:
                seg_idx = torch.randint(0, len(segments), (1,)).item()
            else:
                seg_idx = 0
            mel = torch.tensor(segments[seg_idx]["log_mel"], dtype=torch.float).unsqueeze(0)
            if self.aug_enabled:
                mel = self._spec_augment(mel)
            mel_stack = mel  # [1, n_mels, T]
            num_segs = 1

        # Tokenize text
        text = self.metadata.get_text(tid)
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        label = self.metadata.get_multi_hot_label(tid)

        return {
            "mel": mel_stack,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": label,
            "num_segments": num_segs,
        }


def cnn_bert_collate_fn(batch):
    """Custom collate for CNNBertDataset with variable segment counts."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    seg_counts = [b["num_segments"] for b in batch]

    mels = [b["mel"] for b in batch]

    if mels[0].dim() == 4:
        # Multi-segment: [S, 1, H, W] → pad to max_segs
        max_segs = max(seg_counts)
        _, C, H, W = mels[0].shape
        B = len(batch)
        padded_mels = torch.zeros(B, max_segs, C, H, W)
        seg_mask = torch.zeros(B, max_segs, dtype=torch.bool)
        for i, (m, sc) in enumerate(zip(mels, seg_counts)):
            padded_mels[i, :sc] = m
            seg_mask[i, :sc] = True
    else:
        # Single segment: [1, H, W]
        padded_mels = torch.stack(mels)
        seg_mask = torch.ones(len(batch), 1, dtype=torch.bool)

    return {
        "mel": padded_mels,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "seg_mask": seg_mask,
    }

