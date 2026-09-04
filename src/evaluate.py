"""
Unified Evaluation Script:
  - Loads trained checkpoints for all tasks
  - Computes multi-label metrics (Macro-F1, Micro-F1, Precision, Recall, AUC-PR)
  - Performs validation threshold tuning
  - Compares all models in an ablation table
  - Generates model comparison bar charts and t-SNE embedding plots
"""
from __future__ import annotations


import argparse
import os
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.utils import (
    get_device,
    load_config,
    resolve_paths,
    load_checkpoint,
    get_genre_labels,
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
from src.baselines import CNNBaseline, RandomBaseline, MajorityBaseline
from src.fusion_model import CrossAttentionFusion, EarlyFusion
from src.contrastive import ContrastiveGNNBert, compute_retrieval_metrics
from src.cnn_fusion import ImprovedCNN, CNNBertFusion
from src.metrics import (
    compute_classification_metrics,
    compute_detailed_per_class_metrics,
    tune_threshold,
    tune_per_class_thresholds,
    apply_thresholds,
    save_metrics,
)
from src.visualize import (
    plot_model_comparison,
    plot_tsne_embeddings,
    plot_confusion_matrix,
    generate_comparison_table,
)


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: str, task: str, config: dict):
    """Evaluate a single checkpoint on test set."""
    device = get_device()
    metadata = FMAMetadata(
        metadata_dir=config["dataset"]["metadata_dir"],
        subset=config["dataset"].get("name", "fma_small").replace("fma_", ""),
    )
    splits = metadata.get_split_track_ids()
    val_ids, test_ids = splits["val"], splits["test"]

    processed_dir = config["dataset"]["processed_dir"]
    graphs_dir = os.path.join(processed_dir, "graphs")
    audio_feat_dir = os.path.join(processed_dir, "audio_features")
    bert_cache_dir = os.path.join(processed_dir, "bert")

    batch_size = config["training"].get("batch_size", 32)

    # 1. Instantiate Model & Datasets
    if task == "bert":
        val_ds = BertDataset(val_ids, metadata, config["bert"]["model_name"], config["bert"]["max_length"], bert_cache_dir)
        test_ds = BertDataset(test_ids, metadata, config["bert"]["model_name"], config["bert"]["max_length"], bert_cache_dir)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        model = BertClassifier.from_config(config).to(device)

    elif task in ("gnn", "gat"):
        val_ds = GraphDataset(val_ids, metadata, graphs_dir)
        test_ds = GraphDataset(test_ids, metadata, graphs_dir)
        val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False)

        in_channels = test_ds[0].x.size(-1)
        if task == "gnn":
            model = GraphSAGEClassifier.from_config(config, in_channels=in_channels).to(device)
        else:
            # Previously omitted num_layers/dropout/heads, so this silently
            # fell back to GATClassifier's constructor defaults. That
            # happens to match config.yaml's current values, but only by
            # coincidence — if the config used to train the checkpoint used
            # different gnn.* values, the reconstructed architecture would
            # mismatch the saved state_dict. Build it from config like
            # train.py does.
            model = GATClassifier(
                in_channels=in_channels,
                hidden_channels=config["gnn"]["hidden_channels"],
                num_classes=config["dataset"]["num_genres"],
                num_layers=config["gnn"]["num_layers"],
                heads=config["gnn"].get("heads", 4),
                dropout=config["gnn"]["dropout"],
            ).to(device)

    elif task == "cnn":
        val_ds = MelDataset(val_ids, metadata, audio_feat_dir)
        test_ds = MelDataset(test_ids, metadata, audio_feat_dir)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        model = CNNBaseline.from_config(config).to(device)

    elif task == "improved_cnn":
        icfg = config.get("improved_cnn", {})
        multi_seg = icfg.get("multi_segment", True)
        val_ds = ImprovedMelDataset(val_ids, metadata, audio_feat_dir,
                                    is_training=False, multi_segment=multi_seg)
        test_ds = ImprovedMelDataset(test_ids, metadata, audio_feat_dir,
                                     is_training=False, multi_segment=multi_seg)
        collate = improved_mel_collate_fn if multi_seg else None
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        model = ImprovedCNN.from_config(config).to(device)

    elif task == "cnn_bert_fusion":
        icfg = config.get("improved_cnn", {})
        multi_seg = icfg.get("multi_segment", True)
        val_ds = CNNBertDataset(val_ids, metadata, audio_feat_dir,
                                config["bert"]["model_name"], config["bert"]["max_length"],
                                is_training=False, multi_segment=multi_seg)
        test_ds = CNNBertDataset(test_ids, metadata, audio_feat_dir,
                                 config["bert"]["model_name"], config["bert"]["max_length"],
                                 is_training=False, multi_segment=multi_seg)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=cnn_bert_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=cnn_bert_collate_fn)
        model = CNNBertFusion.from_config(config).to(device)

    elif task in ("fusion", "early_fusion"):
        val_ds = FusionDataset(val_ids, metadata, graphs_dir, config["bert"]["model_name"], config["bert"]["max_length"], bert_cache_dir)
        test_ds = FusionDataset(test_ids, metadata, graphs_dir, config["bert"]["model_name"], config["bert"]["max_length"], bert_cache_dir)
        val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False)

        in_channels = test_ds[0].x.size(-1)
        if task == "fusion":
            model = CrossAttentionFusion.from_config(config, gnn_in_channels=in_channels).to(device)
        else:
            model = EarlyFusion.from_config(config, gnn_in_channels=in_channels).to(device)

    elif task == "contrastive":
        test_ds = ContrastiveDataset(test_ids, metadata, graphs_dir, config["bert"]["model_name"], config["bert"]["max_length"])
        test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False)
        in_channels = test_ds[0].x.size(-1)
        model = ContrastiveGNNBert.from_config(config, gnn_in_channels=in_channels).to(device)
    else:
        raise ValueError(f"Unknown task: {task}")

    # 2. Load Checkpoint weights
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    if task == "contrastive":
        all_audio, all_text = [], []
        for batch in test_loader:
            batch = batch.to(device)
            a_emb, t_emb = model(batch.input_ids, batch.attention_mask, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_audio.append(a_emb)
            all_text.append(t_emb)
        all_audio = torch.cat(all_audio, dim=0)
        all_text = torch.cat(all_text, dim=0)
        metrics = compute_retrieval_metrics(all_audio, all_text)
        return metrics

    # 3. Validation threshold tuning
    val_probs, val_targets = [], []
    for batch in val_loader:
        if task == "bert":
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            targets = batch["label"]
        elif task in ("gnn", "gat"):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y
        elif task == "cnn":
            logits = model(batch[0].to(device))
            targets = batch[1]
        elif task == "improved_cnn":
            if len(batch) == 3:
                mels, targets, seg_mask = batch[0].to(device), batch[1], batch[2].to(device)
                logits = model(mels, seg_mask=seg_mask)
            else:
                logits = model(batch[0].to(device))
                targets = batch[1]
        elif task == "cnn_bert_fusion":
            mels = batch["mel"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"]
            seg_mask = batch["seg_mask"].to(device)
            logits = model(mels, input_ids, attention_mask, seg_mask=seg_mask)
        elif task in ("fusion", "early_fusion"):
            batch = batch.to(device)
            logits = model(batch.input_ids, batch.attention_mask, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y

        if targets.dim() == 1 or targets.shape != logits.shape:
            targets = targets.view(logits.size(0), -1)

        val_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        val_targets.append(targets.detach().cpu().numpy())

    val_probs = np.vstack(val_probs)
    val_targets = np.vstack(val_targets)
    best_thresh, _ = tune_threshold(val_targets, val_probs)
    per_class_thresh, _ = tune_per_class_thresholds(val_targets, val_probs)

    # 4. Test evaluation
    test_probs, test_targets = [], []
    for batch in test_loader:
        if task == "bert":
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            targets = batch["label"]
        elif task in ("gnn", "gat"):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y
        elif task == "cnn":
            logits = model(batch[0].to(device))
            targets = batch[1]
        elif task == "improved_cnn":
            if len(batch) == 3:
                mels, targets, seg_mask = batch[0].to(device), batch[1], batch[2].to(device)
                logits = model(mels, seg_mask=seg_mask)
            else:
                logits = model(batch[0].to(device))
                targets = batch[1]
        elif task == "cnn_bert_fusion":
            mels = batch["mel"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"]
            seg_mask = batch["seg_mask"].to(device)
            logits = model(mels, input_ids, attention_mask, seg_mask=seg_mask)
        elif task in ("fusion", "early_fusion"):
            batch = batch.to(device)
            logits = model(batch.input_ids, batch.attention_mask, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y

        if targets.dim() == 1 or targets.shape != logits.shape:
            targets = targets.view(logits.size(0), -1)

        test_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        test_targets.append(targets.detach().cpu().numpy())

    test_probs = np.vstack(test_probs)
    test_targets = np.vstack(test_targets)
    test_preds = apply_thresholds(test_probs, per_class_thresh)

    metrics = compute_classification_metrics(test_targets, test_preds, test_probs)
    metrics["optimal_threshold"] = float(best_thresh)
    metrics["optimal_per_class_thresholds"] = [float(t) for t in per_class_thresh]
    metrics["per_class_breakdown"] = compute_detailed_per_class_metrics(
        test_targets, test_preds, test_probs, metadata.genre_labels
    )
    return metrics, test_probs, test_targets


def evaluate_all_and_compare(config: dict, results_dir: str | None = None):
    """Run full comparative ablation study across all available models."""
    if results_dir is None:
        results_dir = config["paths"]["results"]

    metadata = FMAMetadata(
        metadata_dir=config["dataset"]["metadata_dir"],
        subset=config["dataset"].get("name", "fma_small").replace("fma_", ""),
    )
    splits = metadata.get_split_track_ids()
    train_ids, test_ids = splits["train"], splits["test"]

    all_results = {}

    # 1. Random Baseline
    train_labels = np.array([metadata.get_multi_hot_label(tid).numpy() for tid in train_ids])
    test_labels = np.array([metadata.get_multi_hot_label(tid).numpy() for tid in test_ids])

    rand_model = RandomBaseline(num_classes=metadata.num_classes)
    rand_preds = rand_model.predict(len(test_labels))
    all_results["Random"] = compute_classification_metrics(test_labels, rand_preds)

    # 2. Majority Baseline
    maj_model = MajorityBaseline(num_classes=metadata.num_classes)
    maj_model.fit(train_labels)
    maj_preds = maj_model.predict(len(test_labels))
    all_results["Majority"] = compute_classification_metrics(test_labels, maj_preds)

    # 3. Search for trained task checkpoints in results/
    task_map = {
        "bert": "BERT-only",
        "gnn": "GNN-only (GraphSAGE)",
        "cnn": "CNN (Mel Spectrogram)",
        "early_fusion": "Early Fusion (Concat)",
        "fusion": "GNN-BERT Cross-Attention",
    }

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for task_key, display_name in task_map.items():
        # Look for checkpoints matching task_best.pt or root results metrics
        ckpts = glob.glob(os.path.join(results_dir, "**", f"{task_key}_best.pt"), recursive=True)
        if ckpts:
            ckpt_path = ckpts[0]
            print(f"Evaluating {display_name} from {ckpt_path}...")
            try:
                metrics, probs, targets = evaluate_checkpoint(ckpt_path, task_key, config)
                all_results[display_name] = metrics
            except Exception as e:
                print(f"  [WARN] Failed to evaluate {task_key}: {e}")
        else:
            # Check if metrics json exists
            metric_files = glob.glob(os.path.join(results_dir, "**", f"{task_key}_metrics.json"), recursive=True)
            if metric_files:
                with open(metric_files[0], "r") as f:
                    all_results[display_name] = json.load(f)

    # 4. Generate comparison table & plots
    table_md = generate_comparison_table(all_results)
    print("\n" + "="*50)
    print(" Ablation Study & Model Comparison Table")
    print("="*50)
    print(table_md)
    print("="*50 + "\n")

    with open(os.path.join(results_dir, "comparison_table.md"), "w") as f:
        f.write(table_md)

    plot_model_comparison(all_results, plots_dir)

    # Save summary json
    with open(os.path.join(results_dir, "all_models_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GNN-BERT Models")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint")
    parser.add_argument("--task", type=str, default="fusion", help="Task for specific checkpoint")
    parser.add_argument("--all", action="store_true", help="Evaluate and compare all trained models")
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolve_paths(cfg)

    if args.all or args.checkpoint is None:
        evaluate_all_and_compare(cfg)
    else:
        # evaluate_checkpoint returns a bare metrics dict for "contrastive",
        # but a (metrics, probs, targets) tuple for every classification
        # task. The old code always treated the result as a plain dict,
        # so for any non-contrastive --task it tried to json.dumps a tuple
        # containing raw numpy arrays and crashed with a TypeError.
        result = evaluate_checkpoint(args.checkpoint, args.task, cfg)
        metrics = result[0] if isinstance(result, tuple) else result
        print(f"Results for {args.task} ({args.checkpoint}):")
        print(json.dumps(metrics, indent=2, default=str))
