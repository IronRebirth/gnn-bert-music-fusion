"""
Utilities: device detection, seeding, config loading, checkpoint management,
experiment directory setup.
"""
from __future__ import annotations

import os
import random
import datetime
from pathlib import Path
from copy import deepcopy
from typing import Optional, Any, Dict, List

import yaml
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress repetitive PyTorch JIT deprecation warnings emitted by PyG during training
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.script is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*torch.jit.*")

import torch.nn.functional as F

# Fix PyTorch MPS bug where scaled_dot_product_attention throws NotImplementedError when dropout_p > 0
_orig_sdpa = F.scaled_dot_product_attention

def _safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    try:
        return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    except NotImplementedError as e:
        if "MPS" in str(e) or (hasattr(query, "device") and query.device.type == "mps"):
            scale_factor = 1 / (query.size(-1) ** 0.5) if scale is None else scale
            attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
            if attn_mask is not None:
                attn_weight = attn_weight + attn_mask
            attn_weight = F.softmax(attn_weight, dim=-1)
            if dropout_p > 0.0:
                attn_weight = F.dropout(attn_weight, p=dropout_p, training=True)
            return torch.matmul(attn_weight, value)
        raise e

F.scaled_dot_product_attention = _safe_sdpa


def get_device() -> torch.device:
    """Auto-detect best available device: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Reproducibility

def set_seed(seed: int = 42):
    """Set random seeds everywhere for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behaviour (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Config

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_config_overrides(config: dict, overrides: dict) -> dict:
    """Recursively merge CLI overrides into config dict."""
    config = deepcopy(config)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key] = merge_config_overrides(config[key], value)
        else:
            config[key] = value
    return config


def resolve_paths(config: dict, project_root: str | None = None):
    """
    Make all relative paths in config absolute relative to *project_root*.
    If running inside Kaggle (where /kaggle/input is read-only), ensures writable
    output directories point to /kaggle/working.
    Modifies config in-place.
    """
    if project_root is None:
        project_root = os.getcwd()
    project_root = Path(project_root)

    # Detect Kaggle environment
    is_kaggle = os.path.exists("/kaggle/working")
    kaggle_working = Path("/kaggle/working") if is_kaggle else project_root

    # Input/read-only path keys
    input_path_keys = [
        ("dataset", "raw_audio_dir"),
        ("dataset", "metadata_dir"),
        ("dataset", "processed_dir"),
        ("dataset", "splits_dir"),
    ]
    for section, key in input_path_keys:
        if section in config and key in config[section]:
            p = Path(config[section][key])
            if not p.is_absolute():
                config[section][key] = str(project_root / p)

    # Output/writable path keys (must NEVER be in /kaggle/input)
    output_path_keys = [
        ("paths", "checkpoints"),
        ("paths", "results"),
        ("paths", "plots"),
    ]
    for section, key in output_path_keys:
        if section in config and key in config[section]:
            p = Path(config[section][key])
            if not p.is_absolute():
                # On Kaggle, output paths should always be under /kaggle/working
                config[section][key] = str(kaggle_working / p)
            elif is_kaggle and str(p).startswith("/kaggle/input"):
                # If an absolute path accidentally pointed into /kaggle/input, reroute to /kaggle/working
                rel = p.name
                config[section][key] = str(kaggle_working / rel)


# Experiment directories

def create_experiment_dir(config: dict, task_name: str) -> str:
    """
    Create a unique experiment directory under results/ and return its path.
    Saves a copy of the config inside.
    """
    results_dir = Path(config["paths"]["results"])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{task_name}_{timestamp}"
    exp_dir = results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)

    # Save config snapshot
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(exp_dir)


# Checkpoints

def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    config: dict,
    extra: dict | None = None,
):
    """Save a resumable training checkpoint safely."""
    # Ensure target directory exists
    target_dir = os.path.dirname(os.path.abspath(path))
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    # Move state dict tensors to CPU before saving to save memory and avoid GPU serialization issues
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ckpt = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model_state,
        "config": config,
    }

    if optimizer is not None:
        try:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        except Exception:
            pass

    if scheduler is not None:
        try:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass

    if extra is not None:
        ckpt.update(extra)

    # Save safely
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device=None):
    """
    Load a checkpoint. Returns (epoch, best_metric, extra_dict).
    Restores model/optimizer/scheduler state in-place.
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", 0.0), ckpt



# FMA metadata helpers

def load_fma_tracks(metadata_dir: str) -> pd.DataFrame:
    """
    Load the FMA tracks.csv with its quirky 3-row header into a usable
    DataFrame indexed by track_id.
    """
    tracks_path = os.path.join(metadata_dir, "tracks.csv")
    # The FMA tracks.csv has a 3-row multi-level header.
    # Row 0: category (album, artist, set, track)
    # Row 1: column name
    # Row 2: "track_id" label row — this is actually the dtype hint row
    df = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    return df


def load_fma_genres(metadata_dir: str) -> dict:
    """Return {genre_id: genre_title} mapping."""
    path = os.path.join(metadata_dir, "genres.csv")
    df = pd.read_csv(path)
    return dict(zip(df["genre_id"], df["title"]))


def get_genre_labels() -> list[str]:
    """
    FMA-small 8 top-level genre labels in alphabetical order.
    This is the fixed label vocabulary for multi-label classification.
    """
    return sorted([
        "Electronic", "Experimental", "Folk", "Hip-Hop",
        "Instrumental", "International", "Pop", "Rock",
    ])


# Misc

def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_device_info():
    """Print GPU / device diagnostics."""
    device = get_device()
    print(f"Device           : {device}")
    if device.type == "cuda":
        print(f"GPU name         : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version     : {torch.version.cuda}")
    print(f"PyTorch version  : {torch.__version__}")
    return device
