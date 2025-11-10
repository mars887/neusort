import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import List, Sequence, Tuple

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
from sorting import farthest_insertion_path


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


def export_clusters(
    paths: Sequence[str],
    feats: np.ndarray,
    result: ClusterResult,
    config: Config,
) -> Tuple[str, int, int]:
    """Persist cluster results to disk and return summary stats."""
    base_dir = config.files.dst_folder
    save_mode = str(getattr(config.clustering, "save_mode", "default")).lower()
    if save_mode not in {"default", "json", "print", "group_filling"}:
        save_mode = "default"
    raw_discard_flag = getattr(config.clustering, "save_discarded", True)
    if isinstance(raw_discard_flag, str):
        lowered_flag = raw_discard_flag.strip().lower()
        save_discarded = lowered_flag not in {"0", "false", "no", "off"}
    else:
        save_discarded = bool(raw_discard_flag)

    if save_mode in {"default", "json", "group_filling"}:
        os.makedirs(base_dir, exist_ok=True)

    # Special mode: flatten clusters into fixed-size groups with filling by discarded
    if save_mode == "group_filling":
        group_size = int(getattr(config.clustering, "group_filling_size", 10) or 10)
        if group_size <= 0:
            group_size = 10
        split_mode = str(getattr(config.clustering, "cluster_splitting_mode", "recluster") or "recluster").lower()
        if split_mode not in {"recluster", "fi"}:
            split_mode = "recluster"

        # Helpers for splitting large clusters
        def _split_cluster_by_fi(cluster: List[int]) -> List[List[int]]:
            if len(cluster) <= group_size:
                return [cluster]
            sub_feats = feats[cluster]
            local_order = farthest_insertion_path(sub_feats, config)
            mapped = [cluster[int(i)] for i in local_order]
            chunks: List[List[int]] = []
            for s in range(0, len(mapped), group_size):
                chunks.append(mapped[s:s + group_size])
            return chunks

        def _split_cluster_by_recluster(cluster: List[int]) -> List[List[int]]:
            if len(cluster) <= group_size:
                return [cluster]

            # Binary search on threshold to enforce max size <= group_size
            base_thr = float(getattr(config.clustering, "threshold", 0.35) or 0.35)
            if base_thr <= 0:
                return _split_cluster_by_fi(cluster)
            lo = max(1e-6, base_thr * 1e-4)
            hi = base_thr
            best: List[List[int]] = []
            found = False

            # Save original settings to restore after temporary modifications
            orig_thr = float(config.clustering.threshold)
            orig_min = int(config.clustering.min_size)
            try:
                # Always allow singletons inside re-cluster step
                for _ in range(14):
                    mid = 0.5 * (lo + hi)
                    config.clustering.threshold = mid
                    config.clustering.min_size = 1
                    sub_feats = feats[cluster]
                    sub_res = cluster_by_distance(sub_feats, config, use_gpu=False)
                    # Map back to original indices; no discards expected (min_size=1)
                    sub_clusters = [[cluster[i] for i in part] for part in sub_res.clusters]
                    max_size = max((len(c) for c in sub_clusters), default=0)
                    if max_size <= group_size and len(sub_clusters) > 0:
                        found = True
                        best = sub_clusters
                        # try to loosen a bit to reduce fragmentation
                        lo = mid
                    else:
                        hi = mid
                    if hi - lo <= base_thr * 1e-4:
                        break
            except Exception:
                # Fallback to FI if anything goes wrong
                best = []
                found = False
            finally:
                config.clustering.threshold = orig_thr
                config.clustering.min_size = orig_min

            if not found or not best:
                return _split_cluster_by_fi(cluster)

            # Ensure resulting parts are chunked by group_size if any part still exceeds (safety)
            out: List[List[int]] = []
            for part in best:
                if len(part) <= group_size:
                    out.append(part)
                else:
                    for s in range(0, len(part), group_size):
                        out.append(part[s:s + group_size])
            return out

        # 1) Prepare clusters: split large ones per chosen mode
        prepared: List[List[int]] = []
        for cl in result.clusters:
            if len(cl) <= group_size:
                prepared.append(list(cl))
            else:
                if split_mode == "fi":
                    prepared.extend(_split_cluster_by_fi(list(cl)))
                else:  # recluster
                    prepared.extend(_split_cluster_by_recluster(list(cl)))

        # Stats: prepared clusters
        if prepared:
            sizes = np.array([len(c) for c in prepared], dtype=np.int32)
            LOGGER.debug(
                f"[group_filling] prepared_clusters={len(prepared)} size_min={int(sizes.min())} "
                f"median={int(np.median(sizes))} max={int(sizes.max())}"
            )
        else:
            LOGGER.debug("[group_filling] no clusters after preparation (all discarded or empty input)")

        from collections import deque
        all_discards = list(result.discarded if save_discarded else [])

        def build_cluster_groups(clusters: List[List[int]], by_size_desc: bool) -> List[List[int]]:
            items = list(clusters)
            if by_size_desc:
                items = sorted(items, key=lambda c: len(c), reverse=True)
            groups: List[List[int]] = []
            current: List[List[int]] = []  # store clusters for the current group (list of lists)
            used_cap = 0
            if not by_size_desc:
                # original order, sequential pack
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
                # first-fit decreasing bin pack by cluster size
                bins: List[Tuple[int, List[List[int]]]] = []  # (used_cap, [clusters])
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
                for cap, arr in bins:
                    groups.append([x for part in arr for x in part])
            return groups

        def fill_groups_with_discards(groups_cluster_only: List[List[int]], discards: List[int]) -> Tuple[List[int], int, int, int]:
            dq = deque(discards)
            seq: List[int] = []
            disc_used = 0
            groups_filled = 0
            groups_incomplete = 0
            for g in groups_cluster_only:
                seq.extend(g)
                gap = max(0, group_size - len(g))
                if gap > 0:
                    groups_filled += 1
                used_here = 0
                while gap > 0 and dq:
                    seq.append(int(dq.popleft()))
                    disc_used += 1
                    used_here += 1
                    gap -= 1
                if gap > 0:
                    groups_incomplete += 1
            # append remaining discards at the end
            if dq:
                seq.extend(list(dq))
            return seq, disc_used, groups_filled, groups_incomplete

        # Phase 1: original-order greedy packing
        groups_seq_order = build_cluster_groups(prepared, by_size_desc=False)
        holes_total = sum(max(0, group_size - len(g)) for g in groups_seq_order)
        need_repack = len(all_discards) < holes_total and len(prepared) > 0
        LOGGER.debug(
            f"[group_filling] phase1 groups={len(groups_seq_order)} holes={holes_total} "
            f"discards_avail={len(all_discards)} repack={'yes' if need_repack else 'no'}"
        )

        if need_repack:
            # Phase 2: FFD repack to reduce holes
            groups_ffd = build_cluster_groups(prepared, by_size_desc=True)
            holes_ffd = sum(max(0, group_size - len(g)) for g in groups_ffd)
            LOGGER.debug(
                f"[group_filling] phase2 FFD groups={len(groups_ffd)} holes={holes_ffd} (improved={holes_ffd < holes_total})"
            )
            target_groups = groups_ffd
        else:
            target_groups = groups_seq_order

        sequence, disc_used, groups_filled, groups_incomplete = fill_groups_with_discards(target_groups, all_discards)
        LOGGER.debug(
            f"[group_filling] filled_groups={groups_filled} groups_incomplete={groups_incomplete} discards_used={disc_used}"
        )

        # Save in the computed order
        if not config.misc.list_only:
            copy_and_rename(paths, sequence, base_dir, config)
        LOGGER.info(
            f"Group-filling output saved to: {base_dir} (groups of {group_size}, split={split_mode})."
        )
        return base_dir, len(result.clusters), len(result.discarded)

    naming_mode = getattr(config.clustering, "naming_mode", "default")
    # JSON/print modes require distance statistics even if the naming mode would not.
    compute_distances = naming_mode in {"distance", "distance_plus"} or save_mode in {"json", "print"}

    clusters_summary: List[dict] = []
    need_alt_output = save_mode in {"json", "print"}
    alt_clusters: List[dict] = []
    file_id_map: dict = {} if need_alt_output else {}

    for idx, cluster in enumerate(result.clusters, start=1):
        cluster_paths = [paths[item] for item in cluster]
        cluster_feats = feats[cluster] if cluster else np.empty((0, feats.shape[1]), dtype=feats.dtype)

        dist_matrix = None
        cluster_avg = None
        per_item_avg = None

        if compute_distances and len(cluster) > 0:
            # Use memory-safe computation; limit full matrix creation to manageable cluster sizes.
            limit = int(getattr(config.clustering, "pairwise_limit", 1200))
            chunk = int(getattr(config.clustering, "distance_chunk_size", 1024))
            dist_matrix, cluster_avg, per_item_avg = _compute_cluster_distances(cluster_feats, limit, chunk)
        elif len(cluster) == 0:
            dist_matrix = np.zeros((0, 0), dtype=np.float32)
            cluster_avg = 0.0
            per_item_avg = np.zeros((0,), dtype=np.float32)

        folder_name = f"cluster_{idx:03d}"
        if naming_mode in {"distance", "distance_plus"} and cluster_avg is not None:
            folder_name = f"{idx:03d}-{_format_distance(cluster_avg)}"

        cluster_dir = os.path.join(base_dir, folder_name)
        distance_names: List[str] = []
        if naming_mode in {"distance", "distance_plus"} and per_item_avg is not None and len(cluster) > 0:
            distance_names = _build_distance_based_names(paths, cluster, per_item_avg)
        saved_names: List[str] = list(distance_names) if distance_names else []

        if save_mode == "default":
            if not config.misc.list_only:
                if distance_names:
                    _copy_with_custom_names(paths, cluster, distance_names, cluster_dir, config)
                    if naming_mode == "distance_plus" and dist_matrix is not None and dist_matrix.size > 0:
                        _write_pairwise_json(cluster_dir, dist_matrix)
                else:
                    copy_and_rename(paths, cluster, cluster_dir, config)
            elif distance_names:
                saved_names = list(distance_names)
        elif distance_names and config.misc.list_only:
            saved_names = list(distance_names)

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

        if (
            naming_mode == "distance_plus"
            and save_mode == "default"
            and not config.misc.list_only
            and len(cluster_paths) > 0
        ):
            cluster_entry["pairwise_json"] = os.path.join(folder_name, "cluster_distances.json")

        clusters_summary.append(cluster_entry)

        if need_alt_output:
            alt_items = []
            for local_idx, original_idx in enumerate(cluster):
                item_path = paths[original_idx]
                if item_path not in file_id_map:
                    file_id_map[item_path] = len(file_id_map) + 1
                file_id = file_id_map[item_path]
                item_entry = {"file_id": file_id}
                if per_item_avg is not None and local_idx < len(per_item_avg):
                    item_entry["average_distance"] = float(per_item_avg[local_idx])
                if saved_names:
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

            alt_clusters.append(cluster_alt)

    if save_mode == "default" and not config.misc.list_only and result.discarded and save_discarded:
        noise_dir = os.path.join(base_dir, "unclustered")
        copy_and_rename(paths, result.discarded, noise_dir, config)
    elif save_mode == "default" and not config.misc.list_only and result.discarded and not save_discarded:
        LOGGER.info("Skipping save of unclustered images because --save_discarded=false was set.")

    summary_path: str

    if save_mode == "default":
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
                "effective_components": int(getattr(config.clustering, "_pca_effective_components", getattr(config.clustering, "pca_components", 0))) if getattr(config.clustering, "pca_enabled", False) else None,
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
    elif save_mode == "json":
        files_section = {path: file_id for path, file_id in sorted(file_id_map.items(), key=lambda item: item[1])}
        payload = {
            "algorithm": getattr(config.clustering, "algorithm", "distance"),
            "pca": {
                "enabled": bool(getattr(config.clustering, "pca_enabled", False)),
                "requested_components": int(getattr(config.clustering, "pca_components", 0)),
                "effective_components": int(getattr(config.clustering, "_pca_effective_components", getattr(config.clustering, "pca_components", 0))) if getattr(config.clustering, "pca_enabled", False) else None,
                "whiten": bool(getattr(config.clustering, "pca_whiten", True)),
            },
            "files": files_section,
            "clusters": alt_clusters,
        }
        summary_path = os.path.join(base_dir, "clusters.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        LOGGER.info(f"Cluster data saved to JSON: {summary_path}")
    else:  # save_mode == "print"
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

    return summary_path, len(result.clusters), len(result.discarded)
