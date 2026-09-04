"""
Graph construction from audio segment features.

Each track becomes a graph where:
  - Nodes  = audio segments (features: mel_mean + chroma_mean)
  - Edges  = temporal adjacency + cosine-similarity above threshold
"""

import os

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / denom)


def build_graph_from_features(
    segments: list[dict],
    similarity_threshold: float = 0.7,
    temporal_weight: float = 1.0,
    include_chroma: bool = True,
) -> Data:
    """
    Build a PyG Data object from a list of segment feature dicts.

    Node features: concatenation of mel_mean and (optionally) chroma_mean.
    Edges: temporal adjacency (|i - j| == 1) and cosine similarity > threshold.
    """
    num_nodes = len(segments)
    if num_nodes == 0:
        raise ValueError("No segments to build graph from")

    # Build node feature matrix
    node_feats = []
    for seg in segments:
        feat = seg["mel_mean"]
        if include_chroma:
            feat = np.concatenate([feat, seg["chroma_mean"]])
        node_feats.append(feat)
    node_feats = np.array(node_feats, dtype=np.float32)  # [N, F]

    src, dst, weights = [], [], []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue

            # Temporal adjacency
            if abs(i - j) == 1:
                src.append(i)
                dst.append(j)
                weights.append(temporal_weight)
            else:
                # Similarity edge
                sim = cosine_similarity(node_feats[i], node_feats[j])
                if sim > similarity_threshold:
                    src.append(i)
                    dst.append(j)
                    weights.append(float(sim))

    if len(src) == 0:
        # Fallback: at least connect consecutive nodes
        for i in range(num_nodes - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])
            weights.extend([temporal_weight, temporal_weight])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float)
    x = torch.tensor(node_feats, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def build_graph_for_track(
    track_id: int,
    features_dir: str,
    graphs_dir: str,
    config: dict,
) -> bool:
    """
    Load features .pt for a track, build graph, save as .pt.
    Returns True on success.
    """
    tid_str = f"{track_id:06d}"
    feat_path = os.path.join(features_dir, f"{tid_str}_features.pt")
    graph_path = os.path.join(graphs_dir, f"{tid_str}_graph.pt")

    if os.path.exists(graph_path):
        return True  # already built

    if not os.path.exists(feat_path):
        return False

    try:
        segments = torch.load(feat_path, weights_only=False)
        graph = build_graph_from_features(
            segments,
            similarity_threshold=config["graph"]["similarity_threshold"],
            temporal_weight=config["graph"]["temporal_edge_weight"],
            include_chroma=config["graph"]["include_chroma_features"],
        )
        torch.save(graph, graph_path)
        return True
    except Exception as e:
        print(f"  [WARN] Graph build failed for {tid_str}: {e}")
        return False


def batch_build_graphs(
    track_ids: list[int],
    features_dir: str,
    graphs_dir: str,
    config: dict,
    desc: str = "Building graphs",
):
    """Build graphs for a batch of tracks."""
    os.makedirs(graphs_dir, exist_ok=True)
    ok, fail = 0, 0
    for tid in tqdm(track_ids, desc=desc):
        if build_graph_for_track(tid, features_dir, graphs_dir, config):
            ok += 1
        else:
            fail += 1
    print(f"Graphs: {ok} succeeded, {fail} failed out of {len(track_ids)}")
