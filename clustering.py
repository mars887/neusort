import json
import math
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import faiss
import numpy as np

from cli import LOGGER
from config import Config
from utils import copy_and_rename


@dataclass
class ClusterResult:
    clusters: List[List[int]]
    discarded: List[int]


def _build_neighbor_sets(feats: np.ndarray, threshold: float, use_gpu: bool) -> List[set]:
    """Return a list where each item contains indices of neighbors within the threshold."""
    n, d = feats.shape
    if n == 0 or threshold <= 0.0:
        return [set() for _ in range(n)]

    feats32 = feats.astype(np.float32, copy=False)
    index = faiss.IndexFlatL2(d)
    if use_gpu:
        try:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
        except Exception as exc:
            LOGGER.info(f"Falling back to CPU FAISS for clustering: {exc}")

    index.add(feats32)
    radius = float(threshold) ** 2
    try:
        lims, _, labels = index.range_search(feats32, radius)
        lims = np.asarray(lims, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
    except TypeError:
        result = faiss.RangeSearchResult(n)
        index.range_search(feats32, radius, result)
        lims = faiss.vector_to_array(result.lims).astype(np.int64, copy=False)
        labels = faiss.vector_to_array(result.labels).astype(np.int64, copy=False)

    neighbors: List[set] = []
    for i in range(n):
        start, end = lims[i], lims[i + 1]
        neighbors.append({int(labels[j]) for j in range(start, end) if int(labels[j]) != i})
    return neighbors


def _assign_clusters(neighbors: Sequence[set], similarity_ratio: float) -> List[List[int]]:
    """Greedy assignment that respects the similarity ratio requirement."""
    clusters: List[List[int]] = []
    similarity_ratio = max(0.0, min(1.0, similarity_ratio))

    for idx, within_threshold in enumerate(neighbors):
        best_cluster = -1
        best_overlap = -1

        for cluster_idx, members in enumerate(clusters):
            overlap = sum(1 for member in members if member in within_threshold)
            required = math.ceil(len(members) * similarity_ratio)
            if overlap >= required and overlap > best_overlap:
                best_overlap = overlap
                best_cluster = cluster_idx

        if best_cluster == -1:
            clusters.append([idx])
        else:
            clusters[best_cluster].append(idx)
    return clusters


def cluster_by_distance(feats: np.ndarray, config: Config, use_gpu: bool) -> ClusterResult:
    """Cluster embeddings based on distance threshold and similarity ratio."""
    threshold = float(config.clustering.threshold)
    similarity_ratio = float(config.clustering.similarity_ratio)
    min_size = int(config.clustering.min_size)

    if threshold <= 0.0:
        LOGGER.error("Clustering threshold must be positive. No clusters will be produced.")
        return ClusterResult(clusters=[], discarded=list(range(feats.shape[0])))

    neighbors = _build_neighbor_sets(feats, threshold, use_gpu)
    raw_clusters = _assign_clusters(neighbors, similarity_ratio)

    valid_clusters: List[List[int]] = []
    discarded: List[int] = []
    for cluster in raw_clusters:
        if len(cluster) >= min_size:
            valid_clusters.append(cluster)
        else:
            discarded.extend(cluster)

    if valid_clusters:
        sizes = np.array([len(cluster) for cluster in valid_clusters], dtype=np.int32)
        LOGGER.info(
            f"Clustering summary: {len(valid_clusters)} clusters, "
            f"size min={int(sizes.min())}, median={int(np.median(sizes))}, max={int(sizes.max())}."
        )
    else:
        LOGGER.info("Clustering summary: no clusters satisfied the minimum size requirement.")

    if discarded:
        LOGGER.info(f"Discarded {len(discarded)} images that did not meet minimum cluster size.")

    return ClusterResult(clusters=valid_clusters, discarded=discarded)


def export_clusters(
    paths: Sequence[str],
    result: ClusterResult,
    config: Config,
) -> Tuple[str, int, int]:
    """Persist cluster results to disk and return summary stats."""
    base_dir = config.files.dst_folder
    os.makedirs(base_dir, exist_ok=True)

    cluster_dirs: List[str] = []
    if not config.misc.list_only:
        for idx, cluster in enumerate(result.clusters, start=1):
            cluster_dir = os.path.join(base_dir, f"cluster_{idx:03d}")
            copy_and_rename(paths, cluster, cluster_dir, config)
            cluster_dirs.append(cluster_dir)

        # if result.discarded:
        #     noise_dir = os.path.join(base_dir, "unclustered")
        #     copy_and_rename(paths, result.discarded, noise_dir, config)
        #     cluster_dirs.append(noise_dir)

    summary = {
        "threshold": config.clustering.threshold,
        "similarity_percent": config.clustering.similarity_percent,
        "min_cluster_size": config.clustering.min_size,
        "cluster_count": len(result.clusters),
        "discarded_count": len(result.discarded),
        "clusters": [
            {
                "id": idx,
                "size": len(cluster),
                "paths": [paths[item] for item in cluster],
            }
            for idx, cluster in enumerate(result.clusters, start=1)
        ],
        "discarded_paths": [paths[item] for item in result.discarded],
    }

    summary_path = os.path.join(base_dir, "clusters.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if config.misc.list_only:
        LOGGER.info(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        LOGGER.info(f"Cluster artifacts saved to: {base_dir}")

    return summary_path, len(result.clusters), len(result.discarded)
