import json
import math
import os
import shutil
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None
import numpy as np
from tqdm.auto import tqdm

from cli import LOGGER
from config import Config
from utils import copy_and_rename
from sorting import farthest_insertion_path, farthest_insertion_path_clustered


@dataclass
class ClusterResult:
    clusters: List[List[int]]
    discarded: List[int]


def _format_distance(value: float) -> str:
    """Return a distance formatted for folder/file naming."""
    return f"{value:.4f}"


def apply_pca_whitening(feats: np.ndarray, n_components: int, whiten: bool, log: bool = True) -> Tuple[np.ndarray, dict]:
    """Apply PCA with optional whitening to features.

    - Centers features to zero mean.
    - Projects onto top `n_components` principal axes.
    - If `whiten` is True, scales components by 1/sqrt(eigenvalue + eps),
      so that Euclidean distance approximates Mahalanobis distance in original space.
    - Returns transformed features (float32) and metadata dict.
    """
    x = feats.astype(np.float32, copy=False)
    n, d = x.shape
    if n == 0 or d == 0:
        return x, {"enabled": False}

    # Determine target dimensionality (cannot exceed rank n-1 or d)
    k_max = max(1, min(d, n - 1))
    k = max(1, min(int(n_components), k_max))

    # Progress indicator
    progress = tqdm(total=3, desc="PCA whitening", disable=not log)

    # 1) Center
    mean = np.mean(x, axis=0, dtype=np.float64)
    xc = (x - mean).astype(np.float32, copy=False)
    progress.update(1)

    # 2) Covariance eigen-decomposition on float64 for stability
    # cov = (xc^T xc) / (n-1)
    cov = (xc.astype(np.float64, copy=False).T @ xc.astype(np.float64, copy=False)) / max(1, (n - 1))
    # eigh returns ascending eigenvalues; reverse for descending order
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    progress.update(1)

    # 3) Projection + optional whitening
    components = evecs[:, :k].astype(np.float32, copy=False)
    vals = evals[:k]
    eps = 1e-8
    if whiten:
        scales = (1.0 / np.sqrt(np.maximum(vals, eps))).astype(np.float32, copy=False)
        z = (xc @ components) * scales
    else:
        z = xc @ components
    z = z.astype(np.float32, copy=False)
    progress.update(1)
    progress.close()

    meta = {
        "enabled": True,
        "original_dim": int(d),
        "n_samples": int(n),
        "components": int(k),
        "whiten": bool(whiten),
    }
    return z, meta


def _compute_cluster_distances(
    cluster_feats: np.ndarray,
    max_items_for_matrix: int,
    chunk_size: int,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute pairwise distances, cluster average distance, and per-item averages.

    Memory-safe implementation:
    - For small clusters (n <= max_items_for_matrix) compute the full n×n matrix with O(n^2) memory using Gram matrix.
    - For large clusters, stream over blocks to compute per-item average distances and cluster average without holding the full matrix.
      In this case, returns an empty distance matrix of shape (0, 0).
    """
    if cluster_feats.size == 0:
        return np.zeros((0, 0), dtype=np.float32), 0.0, np.zeros((0,), dtype=np.float32)

    X = cluster_feats.astype(np.float32, copy=False)
    n, d = X.shape
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32), 0.0, np.zeros((n,), dtype=np.float32)

    # Small clusters: full matrix via Gram trick
    if 0 <= max_items_for_matrix and n <= max_items_for_matrix:
        # Compute squared norms and Gram matrix
        sq = np.sum(X.astype(np.float64, copy=False) ** 2.0, axis=1)
        G = (X.astype(np.float64, copy=False) @ X.astype(np.float64, copy=False).T)
        D2 = sq[:, None] + sq[None, :] - 2.0 * G
        np.maximum(D2, 0.0, out=D2)
        D = np.sqrt(D2, dtype=np.float64).astype(np.float32)
        upper = D[np.triu_indices(n, k=1)]
        cluster_avg = float(np.mean(upper)) if upper.size > 0 else 0.0
        per_item_avg = (D.sum(axis=1) / (n - 1)).astype(np.float32, copy=False)
        return D.astype(np.float32, copy=False), cluster_avg, per_item_avg

    # Large clusters: stream by blocks to avoid O(n^2 * d) memory
    bs = max(64, int(chunk_size))
    sq = np.sum(X.astype(np.float64, copy=False) ** 2.0, axis=1)
    per_sum = np.zeros(n, dtype=np.float64)
    total_upper = 0.0

    # Progress over block pairs roughly proportional to (n/bs)^2
    steps_i = (n + bs - 1) // bs
    show_progress = True  # caller decides via surrounding config; here we always show
    progress = tqdm(total=steps_i, desc="Cluster distances (stream)")

    for i0 in range(0, n, bs):
        i1 = min(n, i0 + bs)
        A = X[i0:i1].astype(np.float64, copy=False)
        a_sq = sq[i0:i1]
        for j0 in range(0, n, bs):
            j1 = min(n, j0 + bs)
            B = X[j0:j1].astype(np.float64, copy=False)
            b_sq = sq[j0:j1]
            G = A @ B.T
            D2 = a_sq[:, None] + b_sq[None, :] - 2.0 * G
            np.maximum(D2, 0.0, out=D2)
            Dblock = np.sqrt(D2, dtype=np.float64)
            # Accumulate per-row sums
            per_sum[i0:i1] += Dblock.sum(axis=1)
            # Accumulate upper-triangular sum for cluster average
            if i0 == j0:
                # only sum strictly upper triangle within block
                m = i1 - i0
                tri_i, tri_j = np.triu_indices(m, k=1)
                total_upper += Dblock[tri_i, tri_j].sum()
            elif i0 < j0:
                total_upper += Dblock.sum()
        if progress:
            progress.update(1)
    if progress:
        progress.close()

    per_item_avg = (per_sum / (n - 1)).astype(np.float32, copy=False)
    denom = n * (n - 1) / 2.0
    cluster_avg = float(total_upper / denom) if denom > 0 else 0.0
    return np.zeros((0, 0), dtype=np.float32), cluster_avg, per_item_avg


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
    index_cpu = faiss.IndexFlatL2(d)
    index_cpu.add(feats32)
    index = index_cpu
    if use_gpu:
        try:
            resources = faiss.StandardGpuResources()
            index_gpu = faiss.index_cpu_to_gpu(resources, 0, index_cpu)
            index = index_gpu
        except Exception as exc:
            LOGGER.info(f"Falling back to CPU FAISS for clustering: {exc}")

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
    except Exception as exc:
        LOGGER.info(f"Range search not available on this FAISS build (using CPU manual fallback): {exc}")
        # Manual CPU fallback: chunked distance computation
        neighbors = [set() for _ in range(n)]
        all_sq = np.sum(feats32.astype(np.float32, copy=False) ** 2.0, axis=1)
        chunk = max(64, min(1024, n))
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            block = feats32[start:end]
            block_sq = all_sq[start:end]
            # distance^2 = ||a||^2 + ||b||^2 - 2 a·b
            d2 = block_sq[:, None] + all_sq[None, :] - 2.0 * (block @ feats32.T)
            np.maximum(d2, 0.0, out=d2)
            within = d2 <= radius
            for i_local, row in enumerate(within):
                i_global = start + i_local
                js = np.where(row)[0]
                for j in js:
                    if j != i_global:
                        neighbors[i_global].add(int(j))
        return neighbors

    neighbors: List[set] = []
    for i in range(n):
        start, end = lims[i], lims[i + 1]
        neighbors.append({int(labels[j]) for j in range(start, end) if int(labels[j]) != i})
    return neighbors


def _assign_clusters(neighbors: Sequence[set], similarity_ratio: float, config: Config) -> List[List[int]]:
    """Greedy assignment that respects the similarity ratio requirement, with progress display."""
    clusters: List[List[int]] = []
    similarity_ratio = max(0.0, min(1.0, similarity_ratio))

    show_progress = config.misc.log_level == "default"
    iterator = range(len(neighbors))
    progress = tqdm(total=len(neighbors), desc="Greedy clustering", disable=not show_progress)

    for idx in iterator:
        within_threshold = neighbors[idx]
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

        if progress:
            progress.update(1)

    if progress:
        progress.close()
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
    raw_clusters = _assign_clusters(neighbors, similarity_ratio, config)

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


def cluster_by_hdbscan(feats: np.ndarray, config: Config) -> ClusterResult:
    """Cluster embeddings using HDBSCAN for higher-accuracy density-based grouping.

    Notes:
    - This method ignores the distance `threshold` and `similarity_ratio` parameters.
    - Uses `min_cluster_size` from config as HDBSCAN's `min_cluster_size`.
    - Metric is set to 'euclidean' since features are L2-normalized.
    - If the `hdbscan` package is unavailable, falls back to the distance-based method.
    """
    n = int(feats.shape[0])
    if n == 0:
        return ClusterResult(clusters=[], discarded=[])

    if hdbscan is None:
        LOGGER.error("HDBSCAN is not installed. Falling back to distance-based clustering.")
        # Fallback to distance-based clustering in CPU mode for safety
        return cluster_by_distance(feats, config, use_gpu=False)

    feats32 = feats.astype(np.float32, copy=False)
    min_cluster_size = max(1, int(config.clustering.min_size))

    # Configure HDBSCAN with a robust default; `min_samples=None` => equals `min_cluster_size`.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="leaf",
    )

    # Show a minimal progress indicator as HDBSCAN runs internally.
    show_progress = config.misc.log_level == "default"
    progress = tqdm(total=1, desc="HDBSCAN fit", disable=not show_progress)
    labels = clusterer.fit_predict(feats32)
    if progress:
        progress.update(1)
        progress.close()

    clusters_dict = {}
    discarded: List[int] = []
    for idx, label in enumerate(labels):
        if int(label) < 0:
            discarded.append(idx)
            continue
        clusters_dict.setdefault(int(label), []).append(idx)

    clusters = list(clusters_dict.values())

    if clusters:
        sizes = np.array([len(c) for c in clusters], dtype=np.int32)
        LOGGER.info(
            f"HDBSCAN summary: {len(clusters)} clusters, "
            f"size min={int(sizes.min())}, median={int(np.median(sizes))}, max={int(sizes.max())}."
        )
    else:
        LOGGER.info("HDBSCAN summary: no clusters were produced.")

    if discarded:
        LOGGER.info(f"HDBSCAN: {len(discarded)} images marked as noise (unclustered).")

    return ClusterResult(clusters=clusters, discarded=discarded)


def cluster_by_dbscan(feats: np.ndarray, config: Config, use_gpu: bool) -> ClusterResult:
    """Cluster embeddings using DBSCAN built on top of FAISS ε-neighborhoods.

    Implementation details:
    - Uses Config.clustering.threshold as eps (radius), and Config.clustering.min_size as min_samples.
    - Neighbor sets are computed via FAISS range_search; then DBSCAN expansion uses a queue.
    - Progress: shows processed points count.
    """
    n = int(feats.shape[0])
    if n == 0:
        return ClusterResult(clusters=[], discarded=[])

    eps = float(config.clustering.threshold)
    if eps <= 0.0:
        LOGGER.error("DBSCAN requires positive threshold (eps). No clusters will be produced.")
        return ClusterResult(clusters=[], discarded=list(range(n)))

    neighbors = _build_neighbor_sets(feats, eps, use_gpu)

    min_samples = max(1, int(config.clustering.min_size))

    UNVISITED = 0
    VISITED = 1
    clustered = -1

    state = np.zeros(n, dtype=np.int8)
    labels = np.full(n, clustered, dtype=np.int32)
    current_label = 0

    show_progress = config.misc.log_level == "default"
    progress = tqdm(total=n, desc="DBSCAN clustering", disable=not show_progress)

    from collections import deque

    for p in range(n):
        if state[p] != UNVISITED:
            continue
        state[p] = VISITED
        Np = neighbors[p]
        # sklearn counts self; our neighbor sets exclude self -> use +1
        if len(Np) + 1 < min_samples:
            # mark as noise by leaving label as -1 for now
            if progress:
                progress.update(1)
            continue
        # Create a new cluster
        labels[p] = current_label
        queue = deque(Np)
        while queue:
            q = queue.popleft()
            if state[q] == UNVISITED:
                state[q] = VISITED
                Nq = neighbors[q]
                if len(Nq) + 1 >= min_samples:
                    queue.extend(Nq)
            if labels[q] == clustered:
                labels[q] = current_label
            # We consider a point processed once it receives a label
        current_label += 1
        if progress:
            progress.update(1)

    if progress:
        # Catch up progress for any unvisited points counted implicitly
        remaining = n - progress.n
        if remaining > 0:
            progress.update(remaining)
        progress.close()

    clusters_dict: dict = {}
    discarded: List[int] = []
    for idx, label in enumerate(labels):
        if label >= 0:
            clusters_dict.setdefault(int(label), []).append(idx)
        else:
            discarded.append(idx)

    clusters = list(clusters_dict.values())

    if clusters:
        sizes = np.array([len(c) for c in clusters], dtype=np.int32)
        LOGGER.info(
            f"DBSCAN summary: {len(clusters)} clusters, "
            f"size min={int(sizes.min())}, median={int(np.median(sizes))}, max={int(sizes.max())}."
        )
    else:
        LOGGER.info("DBSCAN summary: no clusters were produced.")

    if discarded:
        LOGGER.info(f"DBSCAN: {len(discarded)} images marked as noise (unclustered).")

    return ClusterResult(clusters=clusters, discarded=discarded)


def cluster_by_graph(feats: np.ndarray, config: Config, use_gpu: bool) -> ClusterResult:
    """Cluster via connected components on the ε-graph built from FAISS neighbors.

    Implementation details:
    - Build ε-neighborhoods with threshold.
    - Symmetrize adjacency to avoid directional artifacts.
    - Compute connected components via DFS/BFS.
    - Progress: shown for symmetrization and component discovery.
    """
    n = int(feats.shape[0])
    if n == 0:
        return ClusterResult(clusters=[], discarded=[])

    eps = float(config.clustering.threshold)
    if eps <= 0.0:
        LOGGER.error("Graph clustering requires positive threshold. No clusters will be produced.")
        return ClusterResult(clusters=[], discarded=list(range(n)))

    neighbors = _build_neighbor_sets(feats, eps, use_gpu)

    # Symmetrize adjacency to form an undirected graph (OR symmetry)
    show_progress = config.misc.log_level == "default"
    progress_sym = tqdm(total=n, desc="Symmetrizing graph", disable=not show_progress)
    adj = [set(s) for s in neighbors]
    for i in range(n):
        for j in neighbors[i]:
            adj[j].add(i)
        if progress_sym:
            progress_sym.update(1)
    if progress_sym:
        progress_sym.close()

    visited = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []

    progress_cc = tqdm(total=n, desc="Graph components", disable=not show_progress)
    from collections import deque
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        comp = []
        queue = deque([i])
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        clusters.append(comp)
        if progress_cc:
            progress_cc.update(len(comp))
    if progress_cc:
        progress_cc.close()

    # Filter clusters by min_size
    min_size = int(config.clustering.min_size)
    valid_clusters: List[List[int]] = []
    discarded: List[int] = []
    for comp in clusters:
        if len(comp) >= min_size:
            valid_clusters.append(comp)
        else:
            discarded.extend(comp)

    if valid_clusters:
        sizes = np.array([len(c) for c in valid_clusters], dtype=np.int32)
        LOGGER.info(
            f"Graph summary: {len(valid_clusters)} clusters, "
            f"size min={int(sizes.min())}, median={int(np.median(sizes))}, max={int(sizes.max())}."
        )
    else:
        LOGGER.info("Graph summary: no clusters satisfied the minimum size requirement.")

    if discarded:
        LOGGER.info(f"Graph: {len(discarded)} images that did not meet minimum cluster size.")

    return ClusterResult(clusters=valid_clusters, discarded=discarded)


def cluster_by_mutual_graph(feats: np.ndarray, config: Config, use_gpu: bool) -> ClusterResult:
    """Cluster via connected components on the mutual ε-graph.

    Implementation details:
    - Build ε-neighborhoods; keep an undirected edge i—j only if i∈N(j) and j∈N(i).
    - This reduces spurious bridges between clusters compared to plain CC on ε-graph.
    - Progress bars cover mutual graph construction and component extraction.
    """
    n = int(feats.shape[0])
    if n == 0:
        return ClusterResult(clusters=[], discarded=[])

    eps = float(config.clustering.threshold)
    if eps <= 0.0:
        LOGGER.error("Mutual graph clustering requires positive threshold. No clusters will be produced.")
        return ClusterResult(clusters=[], discarded=list(range(n)))

    neighbors = _build_neighbor_sets(feats, eps, use_gpu)

    show_progress = config.misc.log_level == "default"
    progress_build = tqdm(total=n, desc="Building mutual graph", disable=not show_progress)

    # Build mutual adjacency: i—j edge if i in N(j) and j in N(i)
    adj = [set() for _ in range(n)]
    for i in range(n):
        Ni = neighbors[i]
        for j in Ni:
            if i in neighbors[j]:
                adj[i].add(j)
                adj[j].add(i)
        if progress_build:
            progress_build.update(1)
    if progress_build:
        progress_build.close()

    visited = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []

    progress_cc = tqdm(total=n, desc="Graph components", disable=not show_progress)
    from collections import deque
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        comp = []
        queue = deque([i])
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        clusters.append(comp)
        if progress_cc:
            progress_cc.update(len(comp))
    if progress_cc:
        progress_cc.close()

    # Filter clusters by min_size
    min_size = int(config.clustering.min_size)
    valid_clusters: List[List[int]] = []
    discarded: List[int] = []
    for comp in clusters:
        if len(comp) >= min_size:
            valid_clusters.append(comp)
        else:
            discarded.extend(comp)

    if valid_clusters:
        sizes = np.array([len(c) for c in valid_clusters], dtype=np.int32)
        LOGGER.info(
            f"Mutual graph summary: {len(valid_clusters)} clusters, "
            f"size min={int(sizes.min())}, median={int(np.median(sizes))}, max={int(sizes.max())}."
        )
    else:
        LOGGER.info("Mutual graph summary: no clusters satisfied the minimum size requirement.")

    if discarded:
        LOGGER.info(f"Mutual graph: {len(discarded)} images that did not meet minimum cluster size.")

    return ClusterResult(clusters=valid_clusters, discarded=discarded)


def _resolve_save_mode(config: Config) -> str:
    raw_mode = str(getattr(config.clustering, "save_mode", "default")).lower()
    if raw_mode not in {"default", "json", "print", "group_filling", "cluster_sort"}:
        return "default"
    return raw_mode


def _resolve_save_discarded(config: Config) -> bool:
    raw_flag = getattr(config.clustering, "save_discarded", True)
    if isinstance(raw_flag, str):
        lowered = raw_flag.strip().lower()
        return lowered not in {"0", "false", "no", "off"}
    return bool(raw_flag)


def _export_group_filling(
    paths: Sequence[str],
    feats: np.ndarray,
    result: ClusterResult,
    config: Config,
    base_dir: str,
    save_discarded: bool,
) -> Tuple[str, int, int]:
    """Handle save_mode=group_filling."""
    group_size = int(getattr(config.clustering, "group_filling_size", 10) or 10)
    if group_size <= 0:
        group_size = 10
    split_mode = str(getattr(config.clustering, "cluster_splitting_mode", "recluster") or "recluster").lower()
    if split_mode not in {"recluster", "fi"}:
        split_mode = "recluster"

    prepared = _prepare_group_filling_clusters(result.clusters, feats, config, group_size, split_mode)
    prepared = _sort_clusters_by_embedding(prepared, feats, config)

    all_discards = list(result.discarded if save_discarded else [])
    groups_seq_order = _build_cluster_groups(prepared, group_size, by_size_desc=False)
    holes_total = sum(max(0, group_size - len(g)) for g in groups_seq_order)
    need_repack = len(all_discards) < holes_total and len(prepared) > 0
    LOGGER.debug(
        f"[group_filling] phase1 groups={len(groups_seq_order)} holes={holes_total} "
        f"discards_avail={len(all_discards)} repack={'yes' if need_repack else 'no'}"
    )

    if need_repack:
        groups_ffd = _build_cluster_groups(prepared, group_size, by_size_desc=True)
        holes_ffd = sum(max(0, group_size - len(g)) for g in groups_ffd)
        LOGGER.debug(
            f"[group_filling] phase2 FFD groups={len(groups_ffd)} holes={holes_ffd} (improved={holes_ffd < holes_total})"
        )
        target_groups = groups_ffd
    else:
        target_groups = groups_seq_order

    use_similar = bool(getattr(config.clustering, "similar_fill", False))
    if use_similar:
        LOGGER.debug("[group_filling] similar_fill=true (centroid-based)")
    sequence, disc_used, groups_filled, groups_incomplete = _fill_groups_with_discards(
        target_groups, all_discards, group_size, feats, use_similar
    )
    LOGGER.debug(
        f"[group_filling] filled_groups={groups_filled} groups_incomplete={groups_incomplete} discards_used={disc_used}"
    )

    if not config.misc.list_only:
        _save_group_sequence(paths, result, sequence, base_dir, config, group_size, split_mode)

    return base_dir, len(result.clusters), len(result.discarded)


def _order_discarded(discarded: List[int], feats: np.ndarray, config: Config) -> List[int]:
    if not discarded:
        return []
    sub_feats = feats[np.array(discarded, dtype=int)]
    order_local = farthest_insertion_path_clustered(sub_feats, config, show_progress=False)
    return [discarded[int(i)] for i in order_local]


def _cluster_embedding(cluster: List[int], feats: np.ndarray) -> np.ndarray:
    if not cluster:
        return np.zeros((feats.shape[1],), dtype=np.float32)
    return np.mean(feats[np.array(cluster, dtype=int)].astype(np.float32, copy=False), axis=0)


def _order_cluster_items(cluster: List[int], feats: np.ndarray, config: Config) -> List[int]:
    if len(cluster) <= 1:
        return list(cluster)
    sub_feats = feats[np.array(cluster, dtype=int)]
    local_order = farthest_insertion_path(sub_feats, config, show_progress=False)
    return [cluster[int(i)] for i in local_order]


def _cluster_neighbor_cost(
    orient: List[int],
    left_idx: Optional[int],
    right_idx: Optional[int],
    feats: np.ndarray,
    use_centroid: bool,
    centroid: Optional[np.ndarray] = None,
) -> float:
    if not orient:
        return 0.0
    left_vec = centroid if use_centroid and centroid is not None else feats[int(orient[0])]
    right_vec = centroid if use_centroid and centroid is not None else feats[int(orient[-1])]

    cost = 0.0
    if left_idx is not None:
        cost += float(np.linalg.norm(feats[int(left_idx)] - left_vec))
    if right_idx is not None:
        cost += float(np.linalg.norm(right_vec - feats[int(right_idx)]))
    if left_idx is not None and right_idx is not None:
        cost -= float(np.linalg.norm(feats[int(left_idx)] - feats[int(right_idx)]))
    return cost


def _compute_cluster_cli(
    items: List[int],
    base_seq: List[int],
    feats: np.ndarray,
    use_centroid: bool,
) -> Tuple[int, List[int], float, np.ndarray]:
    base_len = len(base_seq)
    centroid = _cluster_embedding(items, feats)
    best_pos = 0
    best_cost = float("inf")
    best_orient = items
    for pos in range(base_len + 1):
        left = base_seq[pos - 1] if pos > 0 else None
        right = base_seq[pos] if pos < base_len else None
        for orient in (items, list(reversed(items))):
            cost = _cluster_neighbor_cost(orient, left, right, feats, use_centroid, centroid)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
                best_orient = orient
    return best_pos, best_orient, best_cost, centroid


def _candidate_starts_for_cluster(desired: int, length: int, current_len: int, group_size: int) -> List[int]:
    if group_size <= 0:
        return [min(desired, current_len)]

    candidates: set[int] = set()
    if length <= group_size:
        groups = {desired // group_size, (desired + length - 1) // group_size}
        for g in groups:
            start_min = g * group_size
            start_max = g * group_size + group_size - length
            if start_max < start_min:
                start_max = start_min
            for s in range(start_min, start_max + 1):
                s_clamped = min(max(0, s), current_len)
                candidates.add(s_clamped)
    else:
        g_start = desired // group_size
        g_end = (desired + length - 1) // group_size
        start_min = g_start * group_size
        start_max = (g_end + 1) * group_size - length
        if start_max < start_min:
            start_max = start_min
        for s in range(start_min, start_max + 1):
            s_clamped = min(max(0, s), current_len)
            candidates.add(s_clamped)

    if not candidates:
        candidates.add(min(desired, current_len))
    return sorted(candidates)


def _placement_cost(
    seq: List[int],
    start: int,
    orient: List[int],
    feats: np.ndarray,
    group_size: int,
    desired: int,
    use_centroid: bool,
    centroid: np.ndarray,
) -> float:
    left = seq[start - 1] if start > 0 else None
    right = seq[start] if start < len(seq) else None
    cost = _cluster_neighbor_cost(orient, left, right, feats, use_centroid, centroid)

    if group_size > 0 and len(orient) > group_size:
        rem = len(orient) % group_size
        if rem == 0 and start % group_size == 0:
            cost -= 0.2
        elif rem < group_size / 2:
            left_partial = group_size - (start % group_size) if (start % group_size) != 0 else group_size
            left_partial = min(left_partial, len(orient))
            remaining = len(orient) - left_partial
            if remaining > 0:
                right_partial = remaining % group_size
                right_partial = right_partial if right_partial != 0 else group_size
            else:
                right_partial = left_partial
            penalty = ((left_partial - right_partial) ** 2) / 150.0 - 0.16
            penalty = min(1.0, max(0.0, penalty))
            cost += penalty

    cost += 0.0005 * abs(start - desired)
    return cost


def _best_insertion_pos(seq: List[int], emb: np.ndarray, feats: np.ndarray) -> Tuple[int, float]:
    best_cost = float("inf")
    best_pos = 0
    n = len(seq)
    for pos in range(n + 1):
        left_idx = seq[pos - 1] if pos > 0 else None
        right_idx = seq[pos] if pos < n else None
        cost = 0.0
        if left_idx is not None:
            cost += float(np.linalg.norm(feats[left_idx] - emb))
        if right_idx is not None:
            cost += float(np.linalg.norm(feats[right_idx] - emb))
        if left_idx is not None and right_idx is not None:
            cost -= float(np.linalg.norm(feats[left_idx] - feats[right_idx]))
        if cost < best_cost:
            best_cost = cost
            best_pos = pos
    return best_pos, best_cost


def _orient_cluster_for_neighbors(order: List[int], feats: np.ndarray, left_idx: Optional[int], right_idx: Optional[int]) -> List[int]:
    if len(order) <= 1:
        return order
    first, last = order[0], order[-1]
    cost_forward = 0.0
    cost_reverse = 0.0
    if left_idx is not None:
        cost_forward += float(np.linalg.norm(feats[left_idx] - feats[first]))
        cost_reverse += float(np.linalg.norm(feats[left_idx] - feats[last]))
    if right_idx is not None:
        cost_forward += float(np.linalg.norm(feats[last] - feats[right_idx]))
        cost_reverse += float(np.linalg.norm(feats[first] - feats[right_idx]))
    return order if cost_forward <= cost_reverse else list(reversed(order))


def _clamp_to_group(idx: int, length: int, group_size: int, total_len: int) -> int:
    if group_size <= 0:
        return max(0, min(idx, total_len))
    group_start = (idx // group_size) * group_size
    lower = group_start
    upper = max(group_start, group_start + group_size - length)
    if upper < lower:
        upper = lower
    clamped = idx
    if clamped < lower:
        clamped = lower
    if clamped > upper:
        clamped = upper
    clamped = max(0, min(clamped, total_len))
    return clamped


def _build_sequence_names(
    sequence: List[int],
    feats: np.ndarray,
    paths: Sequence[str],
    cluster_id_map: Dict[int, int],
    naming_mode: str,
) -> List[str]:
    total = len(sequence)
    num_digits = len(str(total - 1)) if total > 0 else 1
    fmt = f"{{:0{num_digits}d}}"
    names: List[str] = []
    for i, idx in enumerate(sequence):
        src = paths[idx]
        ext = os.path.splitext(src)[1].lower()
        basename = os.path.splitext(os.path.basename(src))[0]
        left_idx = sequence[i - 1] if i > 0 else None
        right_idx = sequence[i + 1] if i + 1 < len(sequence) else None
        d_left = _format_distance(float(np.linalg.norm(feats[idx] - feats[left_idx]))) if left_idx is not None else "0.0000"
        d_right = _format_distance(float(np.linalg.norm(feats[idx] - feats[right_idx]))) if right_idx is not None else "0.0000"
        cluster_id = int(cluster_id_map.get(int(idx), 0))
        if naming_mode == "distance":
            names.append(f"{fmt.format(i)}_{d_left}-{d_right}_{basename}{ext}")
        elif naming_mode == "distance_plus":
            names.append(f"{fmt.format(i)}_{cluster_id}_{d_left}-{d_right}_{basename}{ext}")
        else:
            names.append(f"{fmt.format(i)}_{basename}{ext}")
    return names


def _export_cluster_sort(
    paths: Sequence[str],
    feats: np.ndarray,
    result: ClusterResult,
    config: Config,
    base_dir: str,
) -> Tuple[str, int, int]:
    group_size = int(getattr(config.clustering, "group_filling_size", 10) or 10)
    if group_size <= 0:
        group_size = 10
    naming_mode = getattr(config.clustering, "naming_mode", "default")
    use_centroid = bool(getattr(config.clustering, "cluster_sort_use_centroid", False))

    # Map original cluster id for naming/contiguity
    cluster_id_map: Dict[int, int] = {}
    for cid, cl in enumerate(result.clusters, start=1):
        for idx in cl:
            cluster_id_map[int(idx)] = int(cid)

    # Внутренняя сортировка всех кластеров
    prepared: List[List[int]] = []
    for cl in result.clusters:
        ordered = _order_cluster_items(list(cl), feats, config)
        prepared.append(ordered)

    # Order discarded (noise) as base chain
    base_seq = _order_discarded(list(result.discarded), feats, config)
    feats_cache = feats.astype(np.float32, copy=False)

    # Compute best insertion pos/orientation on base_seq (without mutation)
    cluster_structs = []
    for pid, cl in enumerate(prepared):
        items = list(cl)
        best_pos, best_oriented, best_cost, centroid = _compute_cluster_cli(items, base_seq, feats_cache, use_centroid)
        cluster_structs.append(
            {
                "pid": pid,
                "items": best_oriented,
                "best_pos": best_pos,
                "best_cost": best_cost,
                "orig_size": len(items),
                "centroid": centroid,
            }
        )

    # Sort clusters by CLi (then cost)
    cluster_structs.sort(key=lambda c: (c["best_pos"], c["best_cost"]))

    # Start from ordered discarded list and insert clusters one by one (shifting positions)
    seq: List[int] = list(base_seq)
    spans: List[Tuple[int, int, int]] = []
    offset = 0

    for cl in cluster_structs:
        desired = int(min(cl["best_pos"] + offset, len(seq)))

        best_start = None
        best_orient = cl["items"]
        best_cost_place = float("inf")
        candidates = _candidate_starts_for_cluster(desired, len(cl["items"]), len(seq), group_size)
        for start in candidates:
            for orient in (cl["items"], list(reversed(cl["items"]))):
                cost = _placement_cost(seq, start, orient, feats_cache, group_size, desired, use_centroid, cl["centroid"])
                if cost < best_cost_place:
                    best_cost_place = cost
                    best_start = start
                    best_orient = orient
                elif cost == best_cost_place and best_start is not None and abs(start - desired) < abs(best_start - desired):
                    best_start = start
                    best_orient = orient
        if best_start is None:
            best_start = len(seq)
        seq[best_start:best_start] = best_orient
        offset += len(best_orient)
        spans.append((cl["pid"], best_start, best_start + len(best_orient) - 1))
        LOGGER.debug(
            f"[cluster_sort] insert pid={cl['pid']} size={len(best_orient)} orig_size={cl['orig_size']} "
            f"best_pos={cl['best_pos']} cost={cl['best_cost']:.4f} insert_at={best_start} "
            f"group={best_start//group_size if group_size>0 else 0}"
        )

    for pid, start, end in spans:
        LOGGER.debug(f"[cluster_sort] span pid={pid} start={start} end={end} len={end-start+1}")

    pos_by_cid: Dict[int, List[int]] = {}
    for pos, idx in enumerate(seq):
        cid = int(cluster_id_map.get(int(idx), 0))
        if cid > 0:
            pos_by_cid.setdefault(cid, []).append(pos)

    for cid, positions in pos_by_cid.items():
        positions.sort()
        span_len = positions[-1] - positions[0] + 1
        if span_len != len(positions):
            LOGGER.debug(
                f"[cluster_sort] non-contiguous cluster_id={cid}: size={len(positions)} span={span_len} "
                f"positions(sample)={positions[:5]}...{positions[-5:] if len(positions)>5 else positions}"
            )

    if not config.misc.list_only:
        names = _build_sequence_names(seq, feats, paths, cluster_id_map, naming_mode)
        _copy_with_custom_names(paths, seq, names, base_dir, config)
    LOGGER.info(
        f"Cluster-sort output saved to: {base_dir} "
        f"(groupsize={group_size}, use_centroid={'yes' if use_centroid else 'no'})."
    )
    return base_dir, len(result.clusters), len(result.discarded)


def _prepare_group_filling_clusters(
    clusters: Sequence[Sequence[int]],
    feats: np.ndarray,
    config: Config,
    group_size: int,
    split_mode: str,
) -> List[List[int]]:
    prepared: List[List[int]] = []
    for idx, cl in enumerate(clusters):
        current = list(cl)
        if len(current) <= group_size:
            prepared.append(current)
        elif split_mode == "fi":
            parts = _split_cluster_by_fi(current, feats, config, group_size)
            prepared.extend(parts)
            LOGGER.debug(
                f"[group_filling] split cluster #{idx} size={len(current)} via fi -> parts={list(map(len, parts))}"
            )
        else:
            parts = _split_cluster_by_recluster(current, feats, config, group_size)
            prepared.extend(parts)
            LOGGER.debug(
                f"[group_filling] split cluster #{idx} size={len(current)} via recluster -> parts={list(map(len, parts))}"
            )

    if prepared:
        sizes = np.array([len(c) for c in prepared], dtype=np.int32)
        LOGGER.debug(
            f"[group_filling] prepared_clusters={len(prepared)} size_min={int(sizes.min())} "
            f"median={int(np.median(sizes))} max={int(sizes.max())}"
        )
    else:
        LOGGER.debug("[group_filling] no clusters after preparation (all discarded or empty input)")

    return prepared


def _sort_clusters_by_embedding(clusters: List[List[int]], feats: np.ndarray, config: Config) -> List[List[int]]:
    if not clusters:
        return clusters
    centroids = []
    for cl in clusters:
        if cl:
            centroids.append(np.mean(feats[np.array(cl, dtype=int)].astype(np.float32, copy=False), axis=0))
        else:
            centroids.append(np.zeros((feats.shape[1],), dtype=np.float32))
    centroids_arr = np.stack(centroids, axis=0).astype(np.float32, copy=False)
    order = farthest_insertion_path(centroids_arr, config, show_progress=False)
    return [clusters[int(i)] for i in order]


def _split_cluster_by_fi(
    cluster: List[int],
    feats: np.ndarray,
    config: Config,
    group_size: int,
) -> List[List[int]]:
    if len(cluster) <= group_size:
        return [list(cluster)]
    sub_feats = feats[cluster]
    local_order = farthest_insertion_path(sub_feats, config, show_progress=False)
    mapped = [cluster[int(i)] for i in local_order]
    return [mapped[s : s + group_size] for s in range(0, len(mapped), group_size)]


def _split_cluster_by_recluster(
    cluster: List[int],
    feats: np.ndarray,
    config: Config,
    group_size: int,
) -> List[List[int]]:
    if len(cluster) <= group_size:
        return [list(cluster)]

    base_thr = float(getattr(config.clustering, "threshold", 0.35) or 0.35)
    if base_thr <= 0:
        return _split_cluster_by_fi(cluster, feats, config, group_size)
    lo = max(1e-6, base_thr * 1e-4)
    hi = base_thr * 1.1
    best: List[List[int]] = []
    found = False

    orig_thr = float(config.clustering.threshold)
    orig_min = int(config.clustering.min_size)
    try:
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            config.clustering.threshold = mid
            config.clustering.min_size = 1
            sub_feats = feats[cluster]
            sub_res = cluster_by_distance(sub_feats, config, use_gpu=False)
            sub_clusters = [[cluster[i] for i in part] for part in sub_res.clusters]
            max_size = max((len(c) for c in sub_clusters), default=0)
            if max_size <= group_size and len(sub_clusters) > 0:
                found = True
                best = sub_clusters
                lo = mid
            else:
                hi = mid
            if hi - lo <= base_thr * 1e-4:
                break
    except Exception:
        best = []
        found = False
    finally:
        config.clustering.threshold = orig_thr
        config.clustering.min_size = orig_min

    if not found or not best:
        return _split_cluster_by_fi(cluster, feats, config, group_size)

    out: List[List[int]] = []
    for part in best:
        if len(part) <= group_size:
            out.append(part)
        else:
            for s in range(0, len(part), group_size):
                out.append(part[s : s + group_size])
    return out


def _build_cluster_groups(clusters: List[List[int]], group_size: int, by_size_desc: bool) -> List[List[int]]:
    items = list(clusters)
    if by_size_desc:
        items = sorted(items, key=lambda c: len(c), reverse=True)
    groups: List[List[int]] = []
    current: List[List[int]] = []
    used_cap = 0
    if not by_size_desc:
        idx = 0
        while idx < len(items):
            if used_cap + len(items[idx]) <= group_size:
                current.append(items[idx])
                used_cap += len(items[idx])
                idx += 1
            else:
                groups.append([x for part in current for x in part])
                current = []
                used_cap = 0
        if current:
            groups.append([x for part in current for x in part])
    else:
        bins: List[Tuple[int, List[List[int]]]] = []
        for cl in items:
            placed = False
            for bi in range(len(bins)):
                cap, arr = bins[bi]
                if cap + len(cl) <= group_size:
                    arr.append(cl)
                    bins[bi] = (cap + len(cl), arr)
                    placed = True
                    break
            if not placed:
                bins.append((len(cl), [cl]))
        for _, arr in bins:
            groups.append([x for part in arr for x in part])
    return groups


def _fill_groups_with_discards(
    groups_cluster_only: List[List[int]],
    discards: List[int],
    group_size: int,
    feats: np.ndarray,
    use_similar: bool,
) -> Tuple[List[int], int, int, int]:
    seq: List[int] = []
    disc_used = 0
    groups_filled = 0
    groups_incomplete = 0

    if not use_similar:
        dq = deque(discards)
        for g in groups_cluster_only:
            seq.extend(g)
            gap = max(0, group_size - len(g))
            if gap > 0:
                groups_filled += 1
            while gap > 0 and dq:
                seq.append(int(dq.popleft()))
                disc_used += 1
                gap -= 1
            if gap > 0:
                groups_incomplete += 1
        if dq:
            seq.extend(list(dq))
        return seq, disc_used, groups_filled, groups_incomplete

    # Similar-fill: for each group, pick nearest discards to group centroid
    avail: List[int] = list(discards)
    for g in groups_cluster_only:
        seq.extend(g)
        gap = max(0, group_size - len(g))
        if gap > 0:
            groups_filled += 1
        if gap > 0 and len(avail) > 0 and len(g) > 0:
            k = min(gap, len(avail))
            g_feats = feats[np.array(g, dtype=int)]
            centroid = np.mean(g_feats.astype(np.float32, copy=False), axis=0)
            A = feats[np.array(avail, dtype=int)].astype(np.float32, copy=False)
            dif = A - centroid
            d2 = np.einsum('ij,ij->i', dif, dif)
            pos = np.argpartition(d2, k - 1)[:k]
            pos_sorted = pos[np.argsort(d2[pos])]
            chosen = [avail[int(p)] for p in pos_sorted]
            seq.extend(int(x) for x in chosen)
            disc_used += len(chosen)
            # remove chosen by index positions within avail
            remove_set = set(int(p) for p in pos_sorted)
            avail = [x for i, x in enumerate(avail) if i not in remove_set]
            gap -= len(chosen)
        if gap > 0:
            groups_incomplete += 1
    if len(avail) > 0:
        seq.extend(avail)
    return seq, disc_used, groups_filled, groups_incomplete


def _save_group_sequence(
    paths: Sequence[str],
    result: ClusterResult,
    sequence: List[int],
    base_dir: str,
    config: Config,
    group_size: int,
    split_mode: str,
) -> None:
    cluster_id_map = {}
    for cid, cl in enumerate(result.clusters, start=1):
        for idx0 in cl:
            cluster_id_map[int(idx0)] = int(cid)

    total_files = len(sequence)
    num_digits = len(str(total_files - 1)) if total_files > 0 else 1
    fmt = f"{{:0{num_digits}d}}"
    custom_names: List[str] = []
    for new_pos, orig_idx in enumerate(sequence):
        src = paths[orig_idx]
        ext = os.path.splitext(src)[1].lower()
        original_name = os.path.splitext(os.path.basename(src))[0]
        cluster_id = int(cluster_id_map.get(int(orig_idx), 0))
        custom_names.append(f"{fmt.format(new_pos)}_{cluster_id}_{original_name}{ext}")

    _copy_with_custom_names(paths, sequence, custom_names, base_dir, config)
    LOGGER.info(f"Group-filling output saved to: {base_dir} (groups of {group_size}, split={split_mode}).")


def _build_cluster_payloads(
    paths: Sequence[str],
    feats: np.ndarray,
    result: ClusterResult,
    config: Config,
    base_dir: str,
    save_mode: str,
    naming_mode: str,
    compute_distances: bool,
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    clusters_summary: List[dict] = []
    need_alt_output = save_mode in {"json", "print"}
    alt_clusters: List[dict] = []
    file_id_map: Dict[str, int] = {} if need_alt_output else {}

    for idx, cluster in enumerate(result.clusters, start=1):
        summary_entry, alt_entry = _process_single_cluster(
            idx,
            cluster,
            paths,
            feats,
            config,
            base_dir,
            save_mode,
            naming_mode,
            compute_distances,
            need_alt_output,
            file_id_map,
        )
        clusters_summary.append(summary_entry)
        if alt_entry is not None:
            alt_clusters.append(alt_entry)

    return clusters_summary, alt_clusters, file_id_map


def _process_single_cluster(
    idx: int,
    cluster: List[int],
    paths: Sequence[str],
    feats: np.ndarray,
    config: Config,
    base_dir: str,
    save_mode: str,
    naming_mode: str,
    compute_distances: bool,
    need_alt_output: bool,
    file_id_map: Dict[str, int],
) -> Tuple[dict, Optional[dict]]:
    cluster_paths = [paths[item] for item in cluster]
    cluster_feats = feats[cluster] if cluster else np.empty((0, feats.shape[1]), dtype=feats.dtype)
    dist_matrix, cluster_avg, per_item_avg = _compute_cluster_stats(cluster_feats, compute_distances, len(cluster), config)

    folder_name = f"cluster_{idx:03d}"
    if naming_mode in {"distance", "distance_plus"} and cluster_avg is not None:
        folder_name = f"{idx:03d}-{_format_distance(cluster_avg)}"

    cluster_dir = os.path.join(base_dir, folder_name)
    distance_names: List[str] = []
    if naming_mode in {"distance", "distance_plus"} and per_item_avg is not None and len(cluster) > 0:
        distance_names = _build_distance_based_names(paths, cluster, per_item_avg)
    saved_names = _save_cluster_files(
        save_mode,
        config.misc.list_only,
        distance_names,
        cluster_dir,
        cluster,
        paths,
        config,
        naming_mode,
        dist_matrix,
    )

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
        if saved_names and local_idx < len(saved_names):
            item_info["file_name"] = saved_names[local_idx]
        items.append(item_info)
    cluster_entry["items"] = items

    if (
        naming_mode == "distance_plus"
        and save_mode == "default"
        and not config.misc.list_only
        and len(cluster_paths) > 0
    ):
        cluster_entry["pairwise_json"] = os.path.join(folder_name, "cluster_distances.json")

    alt_entry = _build_alt_cluster_entry(
        idx,
        cluster,
        paths,
        per_item_avg,
        saved_names,
        need_alt_output,
        file_id_map,
        cluster_avg,
        naming_mode,
        dist_matrix,
    )
    return cluster_entry, alt_entry


def _compute_cluster_stats(
    cluster_feats: np.ndarray,
    compute_distances: bool,
    cluster_size: int,
    config: Config,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    dist_matrix: Optional[np.ndarray] = None
    cluster_avg: Optional[float] = None
    per_item_avg: Optional[np.ndarray] = None

    if compute_distances and cluster_size > 0:
        limit = int(getattr(config.clustering, "pairwise_limit", 1200))
        chunk = int(getattr(config.clustering, "distance_chunk_size", 1024))
        dist_matrix, cluster_avg, per_item_avg = _compute_cluster_distances(cluster_feats, limit, chunk)
    elif cluster_size == 0:
        dist_matrix = np.zeros((0, 0), dtype=np.float32)
        cluster_avg = 0.0
        per_item_avg = np.zeros((0,), dtype=np.float32)

    return dist_matrix, cluster_avg, per_item_avg


def _save_cluster_files(
    save_mode: str,
    list_only: bool,
    distance_names: List[str],
    cluster_dir: str,
    cluster: List[int],
    paths: Sequence[str],
    config: Config,
    naming_mode: str,
    dist_matrix: Optional[np.ndarray],
) -> List[str]:
    if save_mode != "default":
        return list(distance_names) if distance_names and list_only else []

    if list_only:
        return list(distance_names) if distance_names else []

    if distance_names:
        _copy_with_custom_names(paths, cluster, distance_names, cluster_dir, config)
        if naming_mode == "distance_plus" and dist_matrix is not None and dist_matrix.size > 0:
            _write_pairwise_json(cluster_dir, dist_matrix)
        return list(distance_names)

    copy_and_rename(paths, cluster, cluster_dir, config)
    return []


def _build_alt_cluster_entry(
    idx: int,
    cluster: List[int],
    paths: Sequence[str],
    per_item_avg: Optional[np.ndarray],
    saved_names: List[str],
    need_alt_output: bool,
    file_id_map: Dict[str, int],
    cluster_avg: Optional[float],
    naming_mode: str,
    dist_matrix: Optional[np.ndarray],
) -> Optional[dict]:
    if not need_alt_output:
        return None

    alt_items = []
    for local_idx, original_idx in enumerate(cluster):
        item_path = paths[original_idx]
        if item_path not in file_id_map:
            file_id_map[item_path] = len(file_id_map) + 1
        file_id = file_id_map[item_path]
        item_entry = {"file_id": file_id}
        if per_item_avg is not None and local_idx < len(per_item_avg):
            item_entry["average_distance"] = float(per_item_avg[local_idx])
        if saved_names and local_idx < len(saved_names):
            item_entry["file_name"] = saved_names[local_idx]
        alt_items.append(item_entry)

    cluster_alt = {
        "id": idx,
        "size": len(cluster),
        "items": alt_items,
    }
    if cluster_avg is not None:
        cluster_alt["average_distance"] = float(cluster_avg)

    if naming_mode == "distance_plus" and dist_matrix is not None and dist_matrix.size > 0 and alt_items:
        pairwise = {}
        for local_idx, item_entry in enumerate(alt_items):
            related = {}
            for other_idx, other_entry in enumerate(alt_items):
                if other_idx == local_idx:
                    continue
                related[str(other_entry["file_id"])] = float(dist_matrix[local_idx, other_idx])
            pairwise[str(item_entry["file_id"])] = related
        cluster_alt["pairwise_distances"] = pairwise

    return cluster_alt


def _copy_discarded_if_needed(
    paths: Sequence[str],
    result: ClusterResult,
    base_dir: str,
    config: Config,
    save_discarded: bool,
) -> None:
    if not result.discarded or config.misc.list_only:
        return

    if save_discarded:
        noise_dir = os.path.join(base_dir, "unclustered")
        copy_and_rename(paths, result.discarded, noise_dir, config)
    else:
        LOGGER.info("Skipping save of unclustered images because --save_discarded=false was set.")


def _write_default_summary(
    base_dir: str,
    paths: Sequence[str],
    result: ClusterResult,
    config: Config,
    naming_mode: str,
    clusters_summary: List[dict],
) -> str:
    summary = {
        "algorithm": getattr(config.clustering, "algorithm", "distance"),
        "threshold": config.clustering.threshold,
        "similarity_percent": config.clustering.similarity_percent,
        "min_cluster_size": config.clustering.min_size,
        "naming_mode": naming_mode,
        "cluster_count": len(result.clusters),
        "discarded_count": len(result.discarded),
        "clusters": clusters_summary,
        "discarded_paths": [paths[item] for item in result.discarded],
        "pca": {
            "enabled": bool(getattr(config.clustering, "pca_enabled", False)),
            "requested_components": int(getattr(config.clustering, "pca_components", 0)),
            "effective_components": int(
                getattr(
                    config.clustering,
                    "_pca_effective_components",
                    getattr(config.clustering, "pca_components", 0),
                )
            )
            if getattr(config.clustering, "pca_enabled", False)
            else None,
            "whiten": bool(getattr(config.clustering, "pca_whiten", True)),
        },
    }
    summary_path = os.path.join(base_dir, "clusters.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if config.misc.list_only:
        LOGGER.info(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        LOGGER.info(f"Cluster artifacts saved to: {base_dir}")
    return summary_path


def _write_json_summary(
    base_dir: str,
    config: Config,
    file_id_map: Dict[str, int],
    alt_clusters: List[dict],
) -> str:
    files_section = {path: file_id for path, file_id in sorted(file_id_map.items(), key=lambda item: item[1])}
    payload = {
        "algorithm": getattr(config.clustering, "algorithm", "distance"),
        "pca": {
            "enabled": bool(getattr(config.clustering, "pca_enabled", False)),
            "requested_components": int(getattr(config.clustering, "pca_components", 0)),
            "effective_components": int(
                getattr(
                    config.clustering,
                    "_pca_effective_components",
                    getattr(config.clustering, "pca_components", 0),
                )
            )
            if getattr(config.clustering, "pca_enabled", False)
            else None,
            "whiten": bool(getattr(config.clustering, "pca_whiten", True)),
        },
        "files": files_section,
        "clusters": alt_clusters,
    }
    summary_path = os.path.join(base_dir, "clusters.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    LOGGER.info(f"Cluster data saved to JSON: {summary_path}")
    return summary_path


def _print_summary(file_id_map: Dict[str, int], alt_clusters: List[dict]) -> str:
    summary_path = "<stdout>"
    print("file,fileId")
    for path, file_id in sorted(file_id_map.items(), key=lambda item: item[1]):
        print(f"{path},{file_id}")
    if alt_clusters:
        print()
    for cluster_alt in alt_clusters:
        fields = [
            f"cluster_{cluster_alt['id']:03d}",
            f"{cluster_alt['size']}",
        ]
        avg_value = cluster_alt.get("average_distance")
        if avg_value is not None:
            fields.append(f"averageDist={avg_value:.6f}")
        for item_entry in cluster_alt["items"]:
            fields.append(str(item_entry["file_id"]))
            avg_item = item_entry.get("average_distance")
            if avg_item is not None:
                fields.append(f"{avg_item:.6f}")
            else:
                fields.append("")
        print(",".join(fields))
    LOGGER.info("Cluster data printed to console in CSV format.")
    return summary_path


def _finalize_cluster_export(
    base_dir: str,
    paths: Sequence[str],
    result: ClusterResult,
    config: Config,
    naming_mode: str,
    save_mode: str,
    save_discarded: bool,
    clusters_summary: List[dict],
    alt_clusters: List[dict],
    file_id_map: Dict[str, int],
) -> str:
    if save_mode == "default":
        _copy_discarded_if_needed(paths, result, base_dir, config, save_discarded)
        return _write_default_summary(base_dir, paths, result, config, naming_mode, clusters_summary)
    if save_mode == "json":
        return _write_json_summary(base_dir, config, file_id_map, alt_clusters)
    return _print_summary(file_id_map, alt_clusters)


def export_clusters(
    paths: Sequence[str],
    feats: np.ndarray,
    result: ClusterResult,
    config: Config,
) -> Tuple[str, int, int]:
    """Persist cluster results to disk and return summary stats."""
    base_dir = config.files.dst_folder
    save_mode = _resolve_save_mode(config)
    save_discarded = _resolve_save_discarded(config)

    if save_mode in {"default", "json", "group_filling", "cluster_sort"}:
        os.makedirs(base_dir, exist_ok=True)

    if save_mode == "group_filling":
        return _export_group_filling(paths, feats, result, config, base_dir, save_discarded)
    if save_mode == "cluster_sort":
        return _export_cluster_sort(paths, feats, result, config, base_dir)

    naming_mode = getattr(config.clustering, "naming_mode", "default")
    compute_distances = naming_mode in {"distance", "distance_plus"} or save_mode in {"json", "print"}

    clusters_summary, alt_clusters, file_id_map = _build_cluster_payloads(
        paths,
        feats,
        result,
        config,
        base_dir,
        save_mode,
        naming_mode,
        compute_distances,
    )

    summary_path = _finalize_cluster_export(
        base_dir,
        paths,
        result,
        config,
        naming_mode,
        save_mode,
        save_discarded,
        clusters_summary,
        alt_clusters,
        file_id_map,
    )
    return summary_path, len(result.clusters), len(result.discarded)
