"""
Unified Training Script for All Models & Tasks:
  - bert          : Task 1 (BERT-only text baseline)
  - gnn           : Task 2 (GraphSAGE audio-only GNN)
  - gat           : Task 2 (GAT variant)
  - cnn           : Audio-only CNN baseline (mel spectrograms)
  - fusion        : Task 3 (GNN-BERT cross-attention fusion)
  - early_fusion  : Task 3 ablation (concatenation baseline)
  - contrastive   : Task 4 (InfoNCE audio-text retrieval)

Features:
  - Automatic device detection (CUDA -> MPS -> CPU)
  - AMP (Automatic Mixed Precision) on CUDA
  - Early stopping with patience
  - Checkpoint saving (best & last) with optimizer/scheduler states
  - Threshold tuning on validation set
  - Macro-F1, Micro-F1, Precision, Recall, AUC-PR tracking
  - Training curve plot generation
  - Configurable CLI overrides
"""
from __future__ import annotations


import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.utils import (
    get_device,
    set_seed,
    load_config,
    merge_config_overrides,
    resolve_paths,
    create_experiment_dir,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    print_device_info,
)
from src.datasets import (
    FMAMetadata,
    BertDataset,
    GraphDataset,
    MelDataset,
    FusionDataset,
    ContrastiveDataset,
    ImprovedMelDataset,
    improved_mel_collate_fn,
    CNNBertDataset,
    cnn_bert_collate_fn,
)
from src.bert_encoder import BertClassifier
from src.gnn_model import GraphSAGEClassifier, GATClassifier
from src.baselines import CNNBaseline
from src.fusion_model import CrossAttentionFusion, EarlyFusion
from src.contrastive import ContrastiveGNNBert, compute_retrieval_metrics
from src.cnn_fusion import ImprovedCNN, CNNBertFusion, FocalLoss
from src.metrics import (
    compute_classification_metrics,
    compute_detailed_per_class_metrics,
    tune_threshold,
    tune_per_class_thresholds,
    apply_thresholds,
    save_metrics,
)
from src.visualize import plot_training_curves


# Training loop helper for classification models

def train_epoch_classification(model, loader, optimizer, criterion, device, scaler=None, is_pyg=False, task_type="standard", max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None and device.type == "cuda")):
            if task_type == "bert":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["label"].to(device)
                logits = model(input_ids, attention_mask)
            elif task_type in ("gnn", "gat"):
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                targets = batch.y.to(device)
            elif task_type == "cnn":
                inputs, targets = batch[0].to(device), batch[1].to(device)
                logits = model(inputs)
            elif task_type == "improved_cnn":
                # Handles both multi-segment (3-tuple) and single-segment (2-tuple)
                if len(batch) == 3:
                    mels, targets, seg_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    logits = model(mels, seg_mask=seg_mask)
                else:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    logits = model(inputs)
            elif task_type == "cnn_bert_fusion":
                mels = batch["mel"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["label"].to(device)
                seg_mask = batch["seg_mask"].to(device)
                logits = model(mels, input_ids, attention_mask, seg_mask=seg_mask)
            elif task_type in ("fusion", "early_fusion"):
                batch = batch.to(device)
                input_ids = batch.input_ids
                attention_mask = batch.attention_mask
                if input_ids.dim() == 1:
                    input_ids = input_ids.view(batch.num_graphs, -1)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.view(batch.num_graphs, -1)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                )
                targets = batch.y.to(device)
            else:
                raise ValueError(f"Unknown task_type: {task_type}")

            if targets.dim() == 1 or targets.shape != logits.shape:
                targets = targets.view(logits.size(0), -1)

            loss = criterion(logits, targets)

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


@torch.no_grad()
def eval_epoch_classification(model, loader, criterion, device, task_type="standard"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds_prob = []
    all_targets = []

    for batch in loader:
        if task_type == "bert":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
        elif task_type in ("gnn", "gat"):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y.to(device)
        elif task_type == "cnn":
            inputs, targets = batch[0].to(device), batch[1].to(device)
            logits = model(inputs)
        elif task_type == "improved_cnn":
            if len(batch) == 3:
                mels, targets, seg_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                logits = model(mels, seg_mask=seg_mask)
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                logits = model(inputs)
        elif task_type == "cnn_bert_fusion":
            mels = batch["mel"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"].to(device)
            seg_mask = batch["seg_mask"].to(device)
            logits = model(mels, input_ids, attention_mask, seg_mask=seg_mask)
        elif task_type in ("fusion", "early_fusion"):
            batch = batch.to(device)
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            if input_ids.dim() == 1:
                input_ids = input_ids.view(batch.num_graphs, -1)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.view(batch.num_graphs, -1)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )
            targets = batch.y.to(device)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        if targets.dim() == 1 or targets.shape != logits.shape:
            targets = targets.view(logits.size(0), -1)

        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds_prob.append(probs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    all_preds_prob = np.vstack(all_preds_prob)
    all_targets = np.vstack(all_targets)
    avg_loss = total_loss / max(1, total_samples)

    return avg_loss, all_preds_prob, all_targets


# Training loop for contrastive model (Task 4)

def train_epoch_contrastive(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        if input_ids.dim() == 1:
            input_ids = input_ids.view(batch.num_graphs, -1)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.view(batch.num_graphs, -1)

        with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None and device.type == "cuda")):
            audio_emb, text_emb = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )
            loss = model.info_nce_loss(audio_emb, text_emb)

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


@torch.no_grad()
def eval_epoch_contrastive(model, loader, device):
    model.eval()
    all_audio_emb = []
    all_text_emb = []

    for batch in loader:
        batch = batch.to(device)
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        if input_ids.dim() == 1:
            input_ids = input_ids.view(batch.num_graphs, -1)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.view(batch.num_graphs, -1)

        audio_emb, text_emb = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )
        all_audio_emb.append(audio_emb)
        all_text_emb.append(text_emb)

    all_audio_emb = torch.cat(all_audio_emb, dim=0)
    all_text_emb = torch.cat(all_text_emb, dim=0)
    loss = model.info_nce_loss(all_audio_emb, all_text_emb).item()
    retrieval_metrics = compute_retrieval_metrics(all_audio_emb, all_text_emb)
    retrieval_metrics["val_loss"] = loss
    return loss, retrieval_metrics


# Main Training Function

def train_model(config: dict, task: str, resume_path: str | None = None, exp_dir: str | None = None):
    set_seed(config["experiment"]["seed"])
    device = get_device()
    print_device_info()

    if exp_dir is None:
        exp_dir = create_experiment_dir(config, task)
    print(f"=== Experiment directory: {exp_dir} ===")

    # 1. Load Metadata & Predefined Splits
    print("Loading FMA metadata...")
    metadata = FMAMetadata(
        metadata_dir=config["dataset"]["metadata_dir"],
        subset=config["dataset"].get("name", "fma_small").replace("fma_", ""),
    )
    splits = metadata.get_split_track_ids()
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    print(f"Dataset split size: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    processed_dir = config["dataset"]["processed_dir"]
    graphs_dir = os.path.join(processed_dir, "graphs")
    audio_feat_dir = os.path.join(processed_dir, "audio_features")
    bert_cache_dir = os.path.join(processed_dir, "bert")

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 2)
    pin_memory = config["training"].get("pin_memory", True) if device.type == "cuda" else False

    # 2. Build Datasets & Loaders
    print(f"Creating datasets for task '{task}'...")
    if task == "bert":
        train_ds = BertDataset(train_ids, metadata, config["bert"]["model_name"], config["bert"]["max_length"], bert_cache_dir)
        val_ds = BertDataset(val_ids, metadata, config["bert"]["model_name"], config["bert"]["max_length"], bert_cache_dir)
        test_ds = BertDataset(test_ids, metadata, config["bert"]["model_name"], config["bert"]["max_length"], bert_cache_dir)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        model = BertClassifier.from_config(config).to(device)

    elif task in ("gnn", "gat"):
        train_ds = GraphDataset(train_ids, metadata, graphs_dir)
        val_ds = GraphDataset(val_ids, metadata, graphs_dir)
        test_ds = GraphDataset(test_ids, metadata, graphs_dir)

        if len(train_ds) == 0:
            raise RuntimeError(f"No graph files found in {graphs_dir}. Run preprocessing first!")

        train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        sample_graph = train_ds[0]
        in_channels = sample_graph.x.size(-1)

        if task == "gnn":
            model = GraphSAGEClassifier.from_config(config, in_channels=in_channels).to(device)
        else:
            model = GATClassifier(
                in_channels=in_channels,
                hidden_channels=config["gnn"]["hidden_channels"],
                num_classes=config["dataset"]["num_genres"],
                num_layers=config["gnn"]["num_layers"],
                heads=config["gnn"].get("heads", 4),
                dropout=config["gnn"]["dropout"],
            ).to(device)

    elif task == "cnn":
        train_ds = MelDataset(train_ids, metadata, audio_feat_dir)
        val_ds = MelDataset(val_ids, metadata, audio_feat_dir)
        test_ds = MelDataset(test_ids, metadata, audio_feat_dir)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        model = CNNBaseline.from_config(config).to(device)

    elif task == "improved_cnn":
        aug_config = config.get("augmentation", {})
        icfg = config.get("improved_cnn", {})
        multi_seg = icfg.get("multi_segment", True)
        rand_seg = aug_config.get("random_segment", False)

        train_ds = ImprovedMelDataset(train_ids, metadata, audio_feat_dir,
                                      is_training=True, augmentation_config=aug_config,
                                      multi_segment=multi_seg, random_segment=rand_seg)
        val_ds = ImprovedMelDataset(val_ids, metadata, audio_feat_dir,
                                    is_training=False, multi_segment=multi_seg)
        test_ds = ImprovedMelDataset(test_ids, metadata, audio_feat_dir,
                                     is_training=False, multi_segment=multi_seg)

        collate = improved_mel_collate_fn if multi_seg else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate)
        model = ImprovedCNN.from_config(config).to(device)

    elif task == "cnn_bert_fusion":
        aug_config = config.get("augmentation", {})
        icfg = config.get("improved_cnn", {})
        multi_seg = icfg.get("multi_segment", True)

        train_ds = CNNBertDataset(train_ids, metadata, audio_feat_dir,
                                  config["bert"]["model_name"], config["bert"]["max_length"],
                                  is_training=True, augmentation_config=aug_config,
                                  multi_segment=multi_seg)
        val_ds = CNNBertDataset(val_ids, metadata, audio_feat_dir,
                                config["bert"]["model_name"], config["bert"]["max_length"],
                                is_training=False, multi_segment=multi_seg)
        test_ds = CNNBertDataset(test_ids, metadata, audio_feat_dir,
                                 config["bert"]["model_name"], config["bert"]["max_length"],
                                 is_training=False, multi_segment=multi_seg)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  collate_fn=cnn_bert_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory,
                                collate_fn=cnn_bert_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory,
                                 collate_fn=cnn_bert_collate_fn)
        model = CNNBertFusion.from_config(config).to(device)

    elif task in ("fusion", "early_fusion"):
        train_ds = FusionDataset(train_ids, metadata, graphs_dir, config["bert"]["model_name"], config["bert"]["max_length"], bert_cache_dir)
        val_ds = FusionDataset(val_ids, metadata, graphs_dir, config["bert"]["model_name"], config["bert"]["max_length"], bert_cache_dir)
        test_ds = FusionDataset(test_ids, metadata, graphs_dir, config["bert"]["model_name"], config["bert"]["max_length"], bert_cache_dir)

        train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        sample_graph = train_ds[0]
        in_channels = sample_graph.x.size(-1)

        if task == "fusion":
            model = CrossAttentionFusion.from_config(config, gnn_in_channels=in_channels).to(device)
        else:
            model = EarlyFusion.from_config(config, gnn_in_channels=in_channels).to(device)

    elif task == "contrastive":
        train_ds = ContrastiveDataset(train_ids, metadata, graphs_dir, config["bert"]["model_name"], config["bert"]["max_length"])
        val_ds = ContrastiveDataset(val_ids, metadata, graphs_dir, config["bert"]["model_name"], config["bert"]["max_length"])
        test_ds = ContrastiveDataset(test_ids, metadata, graphs_dir, config["bert"]["model_name"], config["bert"]["max_length"])

        train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        sample_graph = train_ds[0]
        in_channels = sample_graph.x.size(-1)
        model = ContrastiveGNNBert.from_config(config, gnn_in_channels=in_channels).to(device)

    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"Model initialized: {model.__class__.__name__} ({count_parameters(model):,} trainable parameters)")

    # 3. Optimizer, Scheduler, Loss, Scaler
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.01))

    # Differential learning rates for models with BERT
    if task == "cnn_bert_fusion" and hasattr(model, "get_parameter_groups"):
        cbf_cfg = config.get("cnn_bert_fusion", {})
        param_groups = model.get_parameter_groups(
            bert_lr=float(cbf_cfg.get("bert_lr", 2e-5)),
            new_lr=float(cbf_cfg.get("new_lr", 3e-4)),
        )
        # Add weight_decay to all groups
        for pg in param_groups:
            pg["weight_decay"] = weight_decay
        optimizer = torch.optim.AdamW(param_groups)
        print(f"  Using differential LR: BERT={cbf_cfg.get('bert_lr', 2e-5)}, New={cbf_cfg.get('new_lr', 3e-4)}")
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    epochs = int(config["training"]["epochs"])
    warmup_epochs = int(config["training"].get("warmup_epochs", 0))

    # Scheduler with warmup support
    if config["training"].get("scheduler") == "cosine" and warmup_epochs > 0:
        # Cosine annealing with linear warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    elif config["training"].get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # Calculate class distribution and pos_weight strictly on the training set
    num_classes = config["dataset"]["num_genres"]
    train_labels = np.array([metadata.get_multi_hot_label(tid).numpy() for tid in train_ids])
    pos_counts = train_labels.sum(axis=0)
    total_train = len(train_labels)
    neg_counts = total_train - pos_counts
    # Compute balanced pos_weight: (neg_counts / max(1, pos_counts)), clamped
    # to a sane maximum. FMA-small genres are fairly balanced, but any run on
    # a filtered/custom split could hit a class with very few (or zero)
    # positives, producing a huge pos_weight (up to `total_train`) that
    # dominates the BCE loss and destabilises training for every other
    # class. Capping it keeps rare-class upweighting useful without letting
    # one class blow up the gradient.
    max_pos_weight = 20.0
    pos_weight = torch.tensor(
        [
            min(max_pos_weight, neg_counts[c] / max(1.0, float(pos_counts[c])))
            for c in range(num_classes)
        ],
        dtype=torch.float,
    )
    print(f"\nCalculated Training Set Class Distribution ({num_classes} genres):")
    for c, gname in enumerate(metadata.genre_labels):
        print(f"  {gname:15s}: Pos={int(pos_counts[c]):4d} | Neg={int(neg_counts[c]):4d} | pos_weight={pos_weight[c]:.2f}")

    # Loss function: BCEWithLogitsLoss + pos_weight (default) or FocalLoss
    loss_type = config.get("training", {}).get("loss", "bce")
    if loss_type == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight.to(device))
        print("  Using Focal Loss (alpha=0.25, gamma=2.0)")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scaler = torch.amp.GradScaler("cuda") if (config["training"].get("use_amp", True) and device.type == "cuda") else None

    # 4. Resume from Checkpoint if requested
    start_epoch = 0
    best_metric = -1.0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        start_epoch, best_metric, _ = load_checkpoint(resume_path, model, optimizer, scheduler, device)

    # 5. Training Loop
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": []}
    patience = config["training"].get("patience", 7)
    patience_counter = 0

    best_checkpoint_path = os.path.join(exp_dir, f"{task}_best.pt")
    last_checkpoint_path = os.path.join(exp_dir, f"{task}_last.pt")

    print(f"\n--- Starting Training ({epochs} epochs, patience={patience}) ---")

    for epoch in range(start_epoch + 1, epochs + 1):
        if task == "contrastive":
            train_loss = train_epoch_contrastive(model, train_loader, optimizer, device, scaler)
            val_loss, val_retrieval = eval_epoch_contrastive(model, val_loader, device)
            primary_val_metric = val_retrieval.get("a2t_R@1", 0.0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | a2t_R@1: {val_retrieval.get('a2t_R@1',0):.3f} | a2t_R@5: {val_retrieval.get('a2t_R@5',0):.3f}")

        else:
            train_loss = train_epoch_classification(model, train_loader, optimizer, criterion, device, scaler, task_type=task, max_grad_norm=1.0)
            val_loss, val_probs, val_targets = eval_epoch_classification(model, val_loader, criterion, device, task_type=task)

            # Validation metrics at default 0.5 threshold
            val_metrics = compute_classification_metrics(val_targets, (val_probs >= 0.5).astype(int), val_probs)
            primary_val_metric = val_metrics["macro_f1"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["val_micro_f1"].append(val_metrics["micro_f1"])

            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f} | Val Micro-F1: {val_metrics['micro_f1']:.4f} | Val AUC-PR: {val_metrics.get('auc_pr_macro',0):.4f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(primary_val_metric)
        else:
            scheduler.step()

        # Save last checkpoint
        save_checkpoint(last_checkpoint_path, model, optimizer, scheduler, epoch, best_metric, config)

        # Check for improvement
        if primary_val_metric > best_metric:
            best_metric = primary_val_metric
            patience_counter = 0
            save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, epoch, best_metric, config)
            print(f"  --> Saved new best checkpoint (Metric: {best_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping triggered after {patience} epochs without improvement]")
                break

    # 6. Final Evaluation on Test Set using Best Checkpoint
    print("\n=== Evaluating Best Model on Test Set ===")
    if os.path.exists(best_checkpoint_path):
        load_checkpoint(best_checkpoint_path, model, device=device)

    plots_dir = os.path.join(exp_dir, "plots")
    plot_training_curves(history, plots_dir, prefix=f"{task}_")

    if task == "contrastive":
        test_loss, test_retrieval = eval_epoch_contrastive(model, test_loader, device)
        final_metrics = test_retrieval
        print("Test Retrieval Metrics:", final_metrics)
    else:
        # 1. Validation evaluation & threshold tuning (Validation set ONLY)
        _, val_probs, val_targets = eval_epoch_classification(model, val_loader, criterion, device, task_type=task)
        
        # Strategy A: Default 0.5 threshold
        val_m_default = compute_classification_metrics(val_targets, (val_probs >= 0.5).astype(int), val_probs)
        
        # Strategy B: Globally tuned threshold on validation set
        best_global_thresh, val_global_f1 = tune_threshold(val_targets, val_probs, metric="macro_f1")
        val_m_global = compute_classification_metrics(val_targets, (val_probs >= best_global_thresh).astype(int), val_probs)

        # Strategy C: Per-class tuned thresholds on validation set
        best_per_class_thresh, val_per_class_f1 = tune_per_class_thresholds(val_targets, val_probs)
        val_m_per_class = compute_classification_metrics(val_targets, apply_thresholds(val_probs, best_per_class_thresh), val_probs)

        print("\n--- Validation Threshold Strategies Comparison ---")
        print(f"  Default (0.500)      : Val Macro-F1 = {val_m_default['macro_f1']:.4f} | Val Micro-F1 = {val_m_default['micro_f1']:.4f} | Val AUC-PR = {val_m_default.get('auc_pr_macro', 0):.4f}")
        print(f"  Global Tuned ({best_global_thresh:.3f}): Val Macro-F1 = {val_m_global['macro_f1']:.4f} | Val Micro-F1 = {val_m_global['micro_f1']:.4f} | Val AUC-PR = {val_m_global.get('auc_pr_macro', 0):.4f}")
        print(f"  Per-Class Tuned      : Val Macro-F1 = {val_m_per_class['macro_f1']:.4f} | Val Micro-F1 = {val_m_per_class['micro_f1']:.4f} | Val AUC-PR = {val_m_per_class.get('auc_pr_macro', 0):.4f}")
        for c, gname in enumerate(metadata.genre_labels):
            print(f"    {gname:15s} threshold: {best_per_class_thresh[c]:.3f}")

        # 2. Final Test Set Evaluation (completely untouched test set)
        test_loss, test_probs, test_targets = eval_epoch_classification(model, test_loader, criterion, device, task_type=task)
        
        test_m_default = compute_classification_metrics(test_targets, (test_probs >= 0.5).astype(int), test_probs)
        test_m_global = compute_classification_metrics(test_targets, (test_probs >= best_global_thresh).astype(int), test_probs)
        test_m_per_class = compute_classification_metrics(test_targets, apply_thresholds(test_probs, best_per_class_thresh), test_probs)

        per_class_breakdown = compute_detailed_per_class_metrics(
            test_targets, apply_thresholds(test_probs, best_per_class_thresh), test_probs, metadata.genre_labels
        )

        print("\n--- Final Test Set Results Across Strategies ---")
        print(f"  [Default 0.500] : Test Macro-F1 = {test_m_default['macro_f1']:.4f} | Test Micro-F1 = {test_m_default['micro_f1']:.4f} | Test AUC-PR = {test_m_default.get('auc_pr_macro', 0):.4f}")
        print(f"  [Global Tuned]  : Test Macro-F1 = {test_m_global['macro_f1']:.4f} | Test Micro-F1 = {test_m_global['micro_f1']:.4f} | Test AUC-PR = {test_m_global.get('auc_pr_macro', 0):.4f}")
        print(f"  [Per-Class Opt] : Test Macro-F1 = {test_m_per_class['macro_f1']:.4f} | Test Micro-F1 = {test_m_per_class['micro_f1']:.4f} | Test AUC-PR = {test_m_per_class.get('auc_pr_macro', 0):.4f}")

        print("\n--- Test Set Detailed Per-Class Breakdown (Per-Class Tuned) ---")
        print(f"  {'Genre':15s} | {'Precision':10s} | {'Recall':10s} | {'F1':10s} | {'PR-AUC':10s} | {'Support':8s}")
        print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
        for gname in metadata.genre_labels:
            m_c = per_class_breakdown[gname]
            print(f"  {gname:15s} | {m_c['precision']:10.4f} | {m_c['recall']:10.4f} | {m_c['f1']:10.4f} | {m_c['pr_auc']:10.4f} | {m_c['support']:8d}")

        # Use a shallow copy to avoid circular references when re‑using metric dicts
        final_metrics = test_m_per_class.copy()
        final_metrics["test_loss"] = test_loss
        final_metrics["val_macro_f1"] = val_m_per_class["macro_f1"]
        final_metrics["val_micro_f1"] = val_m_per_class["micro_f1"]
        final_metrics["val_auc_pr"] = val_m_per_class.get("auc_pr_macro", 0.0)
        final_metrics["test_macro_f1"] = test_m_per_class["macro_f1"]
        final_metrics["test_micro_f1"] = test_m_per_class["micro_f1"]
        final_metrics["test_auc_pr"] = test_m_per_class.get("auc_pr_macro", 0.0)
        final_metrics["optimal_global_threshold"] = float(best_global_thresh)
        final_metrics["optimal_per_class_thresholds"] = [float(t) for t in best_per_class_thresh]
        final_metrics["per_class_breakdown"] = per_class_breakdown
        final_metrics["strategies"] = {
            "default_0.5": {"val": val_m_default.copy(), "test": test_m_default.copy()},
            "global_tuned": {"val": val_m_global.copy(), "test": test_m_global.copy(), "threshold": float(best_global_thresh)},
            "per_class_tuned": {"val": val_m_per_class.copy(), "test": test_m_per_class.copy(), "thresholds": [float(t) for t in best_per_class_thresh]},
        }

        # Save predictions CSV
        preds_csv_path = os.path.join(exp_dir, "predictions.csv")
        np.savetxt(preds_csv_path, test_probs, delimiter=",", header=",".join(metadata.genre_labels), comments="")

    metrics_json_path = os.path.join(exp_dir, "metrics.json")
    save_metrics(final_metrics, metrics_json_path)

    # Also copy to root results/ if appropriate
    root_metrics_path = os.path.join(config["paths"]["results"], f"{task}_metrics.json")
    save_metrics(final_metrics, root_metrics_path)

    # Also copy the best/last checkpoints into the canonical checkpoints/
    # directory. Previously checkpoints only ever lived under
    # results/<task>_<timestamp>/, but the README documents
    # "checkpoints/<task>_best.pt" as a stable location, and both
    # notebooks/demo_context.ipynb and scripts/local_train.ipynb's
    # --skip-existing logic look for exactly that path — so without this
    # copy they would never find a trained model.
    canonical_ckpt_dir = config["paths"]["checkpoints"]
    os.makedirs(canonical_ckpt_dir, exist_ok=True)
    if os.path.exists(best_checkpoint_path):
        shutil.copy2(best_checkpoint_path, os.path.join(canonical_ckpt_dir, f"{task}_best.pt"))
    if os.path.exists(last_checkpoint_path):
        shutil.copy2(last_checkpoint_path, os.path.join(canonical_ckpt_dir, f"{task}_last.pt"))

    print(f"Saved all results & checkpoints to {exp_dir}")
    print(f"Best/last checkpoints also copied to {canonical_ckpt_dir}")
    return final_metrics, exp_dir


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Trainer for GNN-BERT Music Context")
    parser.add_argument("--task", type=str, required=True, choices=["bert", "gnn", "gat", "cnn", "improved_cnn", "cnn_bert_fusion", "fusion", "early_fusion", "contrastive"], help="Task to train")
    parser.add_argument("--loss", type=str, default=None, choices=["bce", "focal"], help="Override loss function")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/mps/cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    # Overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.loss is not None:
        cfg["training"]["loss"] = args.loss

    resolve_paths(cfg)
    train_model(cfg, task=args.task, resume_path=args.resume)
