import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import faiss
import numpy as np
from tqdm.auto import tqdm

from cli import LOGGER
from config import Config
from utils import copy_and_rename


@dataclass
class ClusterResult:
    clusters: List[List[int]]
    discarded: List[int]


def _format_distance(value: float) -> str:
    """Return a distance formatted for folder/file naming."""
    return f"{value:.4f}"


def _compute_cluster_distances(cluster_feats: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute pairwise distances, cluster average distance, and per-item averages."""
    if cluster_feats.size == 0:
        return np.zeros((0, 0), dtype=np.float32), 0.0, np.zeros((0,), dtype=np.float32)

    feats32 = cluster_feats.astype(np.float32, copy=False)
    diff = feats32[:, None, :] - feats32[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    size = dist_matrix.shape[0]

    if size <= 1:
        per_item_avg = np.zeros((size,), dtype=np.float32)
        cluster_avg = 0.0
    else:
        upper = dist_matrix[np.triu_indices(size, k=1)]
        cluster_avg = float(np.mean(upper)) if upper.size > 0 else 0.0
        per_item_avg = dist_matrix.sum(axis=1) / (size - 1)

    return dist_matrix, float(cluster_avg), per_item_avg.astype(np.float32, copy=False)


def _build_distance_based_names(paths: Sequence[str], indices: Sequence[int], per_item_avg: np.ndarray) -> List[str]:
    """Create filename stubs that include index within cluster and average distance."""
    total = len(indices)
    digit_count = len(str(total - 1)) if total > 0 else 1
    names: List[str] = []
    for local_idx, original_idx in enumerate(indices):
        ext = os.path.splitext(paths[original_idx])[1].lower()
        stub = f"{str(local_idx).zfill(digit_count)}-{_format_distance(float(per_item_avg[local_idx]))}"
        names.append(f"{stub}{ext}")
    return names


def _copy_with_custom_names(
    paths: Sequence[str],
    indices: Sequence[int],
    names: Sequence[str],
    dst_folder: str,
    config: Config,
) -> None:
    """Copy files using provided names, optionally showing progress."""
    os.makedirs(dst_folder, exist_ok=True)
    total = len(indices)
    progress = None
    if config.misc.log_level == "default" and total > 0:
        desc = f"Copying cluster to '{os.path.basename(dst_folder)}'"
        progress = tqdm(total=total, desc=desc)

    for original_idx, target_name in zip(indices, names):
        src_path = paths[original_idx]
        dst_path = os.path.join(dst_folder, target_name)
        shutil.copy2(src_path, dst_path)
        if progress:
            progress.update(1)

    if progress:
        progress.close()


def _write_pairwise_json(cluster_dir: str, dist_matrix: np.ndarray) -> str:
    """Persist pairwise distance information using cluster-local IDs."""
    file_path = os.path.join(cluster_dir, "cluster_distances.json")
    mapping = {}
    size = dist_matrix.shape[0]
    for i in range(size):
        inner = {
            str(j): float(dist_matrix[i, j])
            for j in range(size)
            if j != i
        }
        mapping[str(i)] = inner
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)
    return file_path


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
    feats: np.ndarray,
    result: ClusterResult,
    config: Config,
) -> Tuple[str, int, int]:
    """Persist cluster results to disk and return summary stats."""
    base_dir = config.files.dst_folder
    os.makedirs(base_dir, exist_ok=True)

    naming_mode = getattr(config.clustering, "naming_mode", "default")
    compute_distances = naming_mode in {"distance", "distance_plus"}

    clusters_summary: List[dict] = []

    for idx, cluster in enumerate(result.clusters, start=1):
        cluster_paths = [paths[item] for item in cluster]
        cluster_feats = feats[cluster] if cluster else np.empty((0, feats.shape[1]), dtype=feats.dtype)

        dist_matrix = None
        cluster_avg = None
        per_item_avg = None

        if compute_distances and len(cluster) > 0:
            dist_matrix, cluster_avg, per_item_avg = _compute_cluster_distances(cluster_feats)
        elif len(cluster) == 0:
            dist_matrix = np.zeros((0, 0), dtype=np.float32)
            cluster_avg = 0.0
            per_item_avg = np.zeros((0,), dtype=np.float32)

        folder_name = f"cluster_{idx:03d}"
        if naming_mode in {"distance", "distance_plus"} and cluster_avg is not None:
            folder_name = f"{idx:03d}-{_format_distance(cluster_avg)}"

        cluster_dir = os.path.join(base_dir, folder_name)
        saved_names: List[str] = []

        if not config.misc.list_only:
            if naming_mode in {"distance", "distance_plus"} and per_item_avg is not None:
                names = _build_distance_based_names(paths, cluster, per_item_avg)
                _copy_with_custom_names(paths, cluster, names, cluster_dir, config)
                saved_names = list(names)
                if naming_mode == "distance_plus" and dist_matrix is not None:
                    _write_pairwise_json(cluster_dir, dist_matrix)
            else:
                copy_and_rename(paths, cluster, cluster_dir, config)
        else:
            if naming_mode in {"distance", "distance_plus"} and per_item_avg is not None:
                saved_names = _build_distance_based_names(paths, cluster, per_item_avg)

        cluster_entry = {
            "id": idx,
            "size": len(cluster),
            "folder_name": folder_name,
            "paths": cluster_paths,
        }
        if cluster_avg is not None:
            cluster_entry["average_distance"] = float(cluster_avg)

        items = []
        for local_idx, original_idx in enumerate(cluster):
            item_info = {"path": paths[original_idx]}
            if per_item_avg is not None and local_idx < len(per_item_avg):
                item_info["average_distance"] = float(per_item_avg[local_idx])
            if saved_names:
                item_info["file_name"] = saved_names[local_idx]
            items.append(item_info)
        cluster_entry["items"] = items

        if naming_mode == "distance_plus" and not config.misc.list_only and len(cluster_paths) > 0:
            cluster_entry["pairwise_json"] = os.path.join(folder_name, "cluster_distances.json")

        clusters_summary.append(cluster_entry)

    if not config.misc.list_only and result.discarded:
        noise_dir = os.path.join(base_dir, "unclustered")
        copy_and_rename(paths, result.discarded, noise_dir, config)

    summary = {
        "threshold": config.clustering.threshold,
        "similarity_percent": config.clustering.similarity_percent,
        "min_cluster_size": config.clustering.min_size,
        "naming_mode": naming_mode,
        "cluster_count": len(result.clusters),
        "discarded_count": len(result.discarded),
        "clusters": clusters_summary,
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
