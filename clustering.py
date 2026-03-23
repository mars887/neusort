import json
import math
import os
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None
try:
    from sklearn.cluster import AgglomerativeClustering, OPTICS as SklearnOPTICS  # type: ignore
except Exception:  # pragma: no cover
    AgglomerativeClustering = None
    SklearnOPTICS = None
import numpy as np
from tqdm.auto import tqdm

from runtime_state import LOGGER
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
    show_progress = LOGGER.lvlp(2)
    progress = tqdm(total=steps_i, desc="Cluster distances (stream)", disable=not show_progress)

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
    if LOGGER.lvlp(2) and total > 0:
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

    show_progress = LOGGER.lvlp(2)
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

    LOGGER.debug(f"[distance] threshold={threshold:.4f} ratio={similarity_ratio:.3f} min_size={min_size}")

    if threshold <= 0.0:
        LOGGER.error("Clustering threshold must be positive. No clusters will be produced.")
        return ClusterResult(clusters=[], discarded=list(range(feats.shape[0])))

    neighbors = _build_neighbor_sets(feats, threshold, use_gpu)
    if LOGGER.lvlp(3) and neighbors:
        counts = np.array([len(n) for n in neighbors], dtype=np.int32)
        LOGGER.debug(
            f"[distance] neighbor stats -> min={int(counts.min())}, mean={float(np.mean(counts)):.2f}, "
            f"max={int(counts.max())}"
        )
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
    LOGGER.debug(f"[hdbscan] samples={n} min_cluster_size={min_cluster_size} dim={feats32.shape[1]}")

    # Configure HDBSCAN with a robust default; `min_samples=None` => equals `min_cluster_size`.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="leaf",
    )

    # Show a minimal progress indicator as HDBSCAN runs internally.
    show_progress = LOGGER.lvlp(2)
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
        LOGGER.debug(f"[hdbscan] noise={len(discarded)}")
    else:
        LOGGER.info("HDBSCAN summary: no clusters were produced.")

    if discarded:
        LOGGER.info(f"HDBSCAN: {len(discarded)} images marked as noise (unclustered).")

    return ClusterResult(clusters=clusters, discarded=discarded)


def cluster_by_agglomerative(feats: np.ndarray, config: Config) -> ClusterResult:
    """Cluster embeddings using agglomerative hierarchical clustering (Ward linkage)."""
    n = int(feats.shape[0])
    if n == 0:
        return ClusterResult(clusters=[], discarded=[])

    if AgglomerativeClustering is None:
        LOGGER.error("scikit-learn is not installed. Falling back to distance-based clustering.")
        return cluster_by_distance(feats, config, use_gpu=False)

    threshold = float(config.clustering.threshold)
    min_size = int(config.clustering.min_size)
    if threshold <= 0.0:
        LOGGER.error("Agglomerative clustering requires positive distance_threshold.")
        return ClusterResult(clusters=[], discarded=list(range(n)))

    feats32 = feats.astype(np.float32, copy=False)
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage="ward",
            metric="euclidean",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage="ward",
            affinity="euclidean",
        )

    labels = model.fit_predict(feats32)

    clusters_dict: Dict[int, List[int]] = {}
    discarded: List[int] = []
    for idx, label in enumerate(labels):
        clusters_dict.setdefault(int(label), []).append(idx)

    valid_clusters: List[List[int]] = []
    for members in clusters_dict.values():
        if len(members) >= min_size:
            valid_clusters.append(members)
        else:
            discarded.extend(members)

    if valid_clusters:
        sizes = np.array([len(c) for c in valid_clusters], dtype=np.int32)
        LOGGER.info(
            f"Agglomerative summary: {len(valid_clusters)} clusters, "
            f"size min={int(sizes.min())}, median={int(np.median(sizes))}, max={int(sizes.max())}."
        )
        LOGGER.debug(f"[agglomerative] label_count={len(clusters_dict)} discarded={len(discarded)}")
    else:
        LOGGER.info("Agglomerative summary: no clusters satisfied the minimum size requirement.")

    if discarded:
        LOGGER.info(f"Agglomerative: {len(discarded)} images discarded for being below min_size.")

    return ClusterResult(clusters=valid_clusters, discarded=discarded)


def cluster_by_optics(feats: np.ndarray, config: Config) -> ClusterResult:
    """Cluster embeddings using OPTICS (Ordering Points To Identify the Clustering Structure)."""
    n = int(feats.shape[0])
    if n == 0:
        return ClusterResult(clusters=[], discarded=[])

    if SklearnOPTICS is None:
        LOGGER.error("scikit-learn is not installed. Falling back to DBSCAN-style clustering.")
        return cluster_by_dbscan(feats, config, use_gpu=False)

    min_samples = max(2, int(config.clustering.min_size))
    max_eps = float(config.clustering.threshold)
    if max_eps <= 0.0:
        max_eps = np.inf

    feats32 = feats.astype(np.float32, copy=False)
    LOGGER.debug(f"[optics] samples={n} min_samples={min_samples} max_eps={max_eps}")
    model = SklearnOPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        metric="euclidean",
        cluster_method="xi",
    )

    labels = model.fit_predict(feats32)

    clusters_dict: Dict[int, List[int]] = {}
    discarded: List[int] = []
    for idx, label in enumerate(labels):
        if int(label) < 0:
            discarded.append(idx)
            continue
        clusters_dict.setdefault(int(label), []).append(idx)

    raw_clusters = list(clusters_dict.values())
    min_size = int(config.clustering.min_size)
    final_clusters: List[List[int]] = []
    for cluster in raw_clusters:
        if len(cluster) >= min_size:
            final_clusters.append(cluster)
        else:
            discarded.extend(cluster)

    if final_clusters:
        sizes = np.array([len(c) for c in final_clusters], dtype=np.int32)
        LOGGER.info(
            f"OPTICS summary: {len(final_clusters)} clusters, "
            f"size min={int(sizes.min())}, median={int(np.median(sizes))}, max={int(sizes.max())}."
        )
    else:
        LOGGER.info("OPTICS summary: no clusters satisfied the minimum size requirement.")

    if discarded:
        LOGGER.info(f"OPTICS: {len(discarded)} images marked as noise or below min_size.")

    return ClusterResult(clusters=final_clusters, discarded=discarded)


def cluster_by_agglomerative_complete(feats: np.ndarray, config: Config) -> ClusterResult:
    """Hierarchical clustering with complete linkage (max-pair distance <= threshold)."""
    n = int(feats.shape[0])
    if n == 0:
        return ClusterResult(clusters=[], discarded=[])

    if AgglomerativeClustering is None:
        LOGGER.error("scikit-learn is not installed. Falling back to distance-based clustering.")
        return cluster_by_distance(feats, config, use_gpu=False)

    threshold = float(config.clustering.threshold)
    min_size = int(config.clustering.min_size)
    if threshold <= 0.0:
        LOGGER.error("Agglomerative (complete) requires positive distance_threshold.")
        return ClusterResult(clusters=[], discarded=list(range(n)))

    feats32 = feats.astype(np.float32, copy=False)
    LOGGER.info("Running Agglomerative Clustering (Complete Linkage)...")
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage="complete",
            metric="euclidean",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage="complete",
            affinity="euclidean",
        )

    labels = model.fit_predict(feats32)

    clusters_dict: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters_dict.setdefault(int(label), []).append(idx)

    valid_clusters: List[List[int]] = []
    discarded: List[int] = []
    for members in clusters_dict.values():
        if len(members) >= min_size:
            valid_clusters.append(members)
        else:
            discarded.extend(members)

    if valid_clusters:
        sizes = np.array([len(c) for c in valid_clusters], dtype=np.int32)
        LOGGER.info(
            f"Agglomerative (complete) summary: {len(valid_clusters)} clusters, "
            f"size min={int(sizes.min())}, median={int(np.median(sizes))}, max={int(sizes.max())}."
        )
        LOGGER.debug(f"[agglomerative_complete] label_count={len(clusters_dict)} discarded={len(discarded)}")
    else:
        LOGGER.info("Agglomerative (complete) summary: no clusters satisfied the minimum size requirement.")

    if discarded:
        LOGGER.info(f"Agglomerative (complete): {len(discarded)} images discarded for being below min_size.")

    return ClusterResult(clusters=valid_clusters, discarded=discarded)


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
    if LOGGER.lvlp(3) and neighbors:
        deg = np.array([len(s) for s in neighbors], dtype=np.int32)
        LOGGER.debug(
            f"[dbscan] eps={eps:.4f} min_samples={min_samples} neighbor deg stats "
            f"min={int(deg.min())} mean={float(np.mean(deg)):.2f} max={int(deg.max())}"
        )

    UNVISITED = 0
    VISITED = 1
    clustered = -1

    state = np.zeros(n, dtype=np.int8)
    labels = np.full(n, clustered, dtype=np.int32)
    current_label = 0

    show_progress = LOGGER.lvlp(2)
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
    if LOGGER.lvlp(3) and neighbors:
        deg = np.array([len(s) for s in neighbors], dtype=np.int32)
        LOGGER.debug(
            f"[cc_graph] eps={eps:.4f} min_size={config.clustering.min_size} "
            f"deg stats min={int(deg.min())} mean={float(np.mean(deg)):.2f} max={int(deg.max())}"
        )

    # Symmetrize adjacency to form an undirected graph (OR symmetry)
    show_progress = LOGGER.lvlp(2)
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
    if LOGGER.lvlp(3) and neighbors:
        deg = np.array([len(s) for s in neighbors], dtype=np.int32)
        LOGGER.debug(
            f"[mutual_graph] eps={eps:.4f} min_size={config.clustering.min_size} "
            f"deg stats min={int(deg.min())} mean={float(np.mean(deg)):.2f} max={int(deg.max())}"
        )

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


def cluster_by_snn(feats: np.ndarray, config: Config, use_gpu: bool) -> ClusterResult:
    """Cluster embeddings using a Shared Nearest Neighbors (SNN) graph."""
    n, d = feats.shape
    if n == 0:
        return ClusterResult(clusters=[], discarded=[])

    min_size = int(config.clustering.min_size)
    k = max(20, min_size * 2)
    if k >= n:
        k = n - 1
    if k < 1:
        if n >= min_size:
            return ClusterResult(clusters=[list(range(n))], discarded=[])
        return ClusterResult(clusters=[], discarded=list(range(n)))

    raw_thr = float(config.clustering.threshold)
    if raw_thr > 1.0:
        shared_thr = int(raw_thr)
    elif raw_thr > 0.0:
        shared_thr = int(k * raw_thr)
    else:
        shared_thr = int(k * 0.15)
    shared_thr = max(1, shared_thr)

    LOGGER.info(f"SNN clustering parameters: k={k}, required_shared={shared_thr}")
    LOGGER.debug(f"[snn] samples={n} dim={d} k={k} shared_thr={shared_thr}")

    feats32 = feats.astype(np.float32, copy=False)
    index_cpu = faiss.IndexFlatL2(d)
    index_cpu.add(feats32)
    index = index_cpu
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        except Exception as exc:
            LOGGER.info(f"SNN: falling back to CPU FAISS: {exc}")

    _, I = index.search(feats32, k + 1)

    neighbors_sets = [set(map(int, I[i, 1:])) for i in range(n)]
    adj: List[List[int]] = [[] for _ in range(n)]

    show_progress = LOGGER.lvlp(2)
    progress = tqdm(total=n, desc="SNN Graph Build", disable=not show_progress)

    for i in range(n):
        i_neighbors = neighbors_sets[i]
        candidates = I[i, 1:]
        for j_raw in candidates:
            j = int(j_raw)
            if j <= i:
                continue
            shared_count = len(i_neighbors.intersection(neighbors_sets[j]))
            if shared_count >= shared_thr:
                adj[i].append(j)
                adj[j].append(i)
        if progress:
            progress.update(1)
    if progress:
        progress.close()

    visited = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []

    progress_cc = tqdm(total=n, desc="SNN Components", disable=not show_progress)
    from collections import deque
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        comp: List[int] = []
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
            f"SNN summary: {len(valid_clusters)} clusters, "
            f"size min={int(sizes.min())}, median={int(np.median(sizes))}, max={int(sizes.max())}."
        )
    else:
        LOGGER.info("SNN summary: no clusters satisfied the minimum size requirement.")

    if discarded:
        LOGGER.info(f"SNN: {len(discarded)} images marked as noise or below min_size.")

    return ClusterResult(clusters=valid_clusters, discarded=discarded)


def cluster_by_rank_mutual(feats: np.ndarray, config: Config, use_gpu: bool) -> ClusterResult:
    """Cluster using reciprocal k-NN (rank-based mutual links)."""
    n, d = feats.shape
    if n == 0:
        return ClusterResult(clusters=[], discarded=[])

    min_size = int(config.clustering.min_size)
    k = max(5, int(min_size * 2) + 2)
    k = min(k, 50)
    if n == 1:
        return ClusterResult(clusters=[list(range(n))] if min_size <= 1 else [], discarded=([] if min_size <= 1 else [0]))
    k = min(k, max(1, n - 1))

    LOGGER.info(f"Rank Mutual Clustering: searching top-{k} neighbors (reciprocal).")
    LOGGER.debug(f"[rank_mutual] samples={n} min_size={min_size} k={k} dim={d}")

    feats32 = feats.astype(np.float32, copy=False)
    index_cpu = faiss.IndexFlatL2(d)
    index_cpu.add(feats32)
    index = index_cpu
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        except Exception as exc:
            LOGGER.info(f"Rank Mutual: falling back to CPU FAISS: {exc}")

    _, I = index.search(feats32, k + 1)

    neighbor_sets = [set(int(x) for x in I[i, 1:]) for i in range(n)]
    adj: List[List[int]] = [[] for _ in range(n)]

    show_progress = LOGGER.lvlp(2)
    progress = tqdm(total=n, desc="Building reciprocal graph", disable=not show_progress)

    for i in range(n):
        candidates = I[i, 1:]
        for neighbor_idx in candidates:
            j = int(neighbor_idx)
            if j <= i:
                continue
            if i in neighbor_sets[j]:
                adj[i].append(j)
                adj[j].append(i)
        if progress:
            progress.update(1)
    if progress:
        progress.close()

    visited = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []
    from collections import deque
    for i in range(n):
        if visited[i]:
            continue
        if not adj[i]:
            visited[i] = True
            continue
        visited[i] = True
        comp: List[int] = []
        queue = deque([i])
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        clusters.append(comp)

    valid_clusters: List[List[int]] = []
    for comp in clusters:
        if len(comp) >= min_size:
            valid_clusters.append(comp)

    clustered_mask = np.zeros(n, dtype=bool)
    if valid_clusters:
        clustered_mask[np.concatenate(valid_clusters)] = True
    discarded = list(np.nonzero(~clustered_mask)[0])

    if valid_clusters:
        sizes = np.array([len(c) for c in valid_clusters], dtype=np.int32)
        LOGGER.info(
            f"Rank Mutual summary: {len(valid_clusters)} clusters, "
            f"size min={int(sizes.min())}, median={int(np.median(sizes))}, max={int(sizes.max())}."
        )
    else:
        LOGGER.info("Rank Mutual summary: no clusters produced.")

    if discarded:
        LOGGER.info(f"Rank Mutual: {len(discarded)} images marked as noise or below min_size.")

    return ClusterResult(clusters=valid_clusters, discarded=discarded)


def cluster_by_adaptive_graph(feats: np.ndarray, config: Config, use_gpu: bool) -> ClusterResult:
    """Two-pass adaptive mutual graph clustering with stricter split for large clusters."""
    n = int(feats.shape[0])
    if n == 0:
        return ClusterResult(clusters=[], discarded=[])

    loose_threshold = float(config.clustering.threshold)
    min_size = int(config.clustering.min_size)
    max_size_before_split = max(20, min_size * 5)
    strict_ratio = 0.85

    if loose_threshold <= 0.0:
        return ClusterResult(clusters=[], discarded=list(range(n)))

    neighbors = _build_neighbor_sets(feats, loose_threshold, use_gpu)
    if LOGGER.lvlp(3) and neighbors:
        deg = np.array([len(s) for s in neighbors], dtype=np.int32)
        LOGGER.debug(
            f"[adaptive_graph] loose={loose_threshold:.4f} strict_ratio={strict_ratio} split_size>{max_size_before_split} "
            f"deg stats min={int(deg.min())} mean={float(np.mean(deg)):.2f} max={int(deg.max())}"
        )
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in neighbors[i]:
            if i in neighbors[j]:
                adj[i].add(j)
                adj[j].add(i)

    visited = np.zeros(n, dtype=bool)
    raw_clusters: List[List[int]] = []
    from collections import deque
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if not adj[i]:
            continue
        comp: List[int] = []
        queue = deque([i])
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        raw_clusters.append(comp)

    final_clusters: List[List[int]] = []
    discarded: List[int] = []
    strict_threshold_sq = (loose_threshold * strict_ratio) ** 2

    LOGGER.info(f"Adaptive Graph: inspecting {len(raw_clusters)} clusters. Split limit={max_size_before_split}.")
    show_progress = LOGGER.lvlp(2)

    for cluster in tqdm(raw_clusters, desc="Adaptive refinement", disable=not show_progress):
        LOGGER.debug(f"[adaptive_graph] cluster_size={len(cluster)}")
        if len(cluster) <= max_size_before_split:
            if len(cluster) >= min_size:
                final_clusters.append(cluster)
            else:
                discarded.extend(cluster)
            continue

        indices = np.array(cluster, dtype=int)
        sub_feats = feats[indices]
        m = len(indices)
        sub_adj: List[List[int]] = [[] for _ in range(m)]

        sf = sub_feats.astype(np.float32, copy=False)
        sq_norms = np.sum(sf ** 2, axis=1)
        d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (sf @ sf.T)
        np.maximum(d2, 0.0, out=d2)

        for i_loc in range(m):
            for j_loc in range(i_loc + 1, m):
                if d2[i_loc, j_loc] <= strict_threshold_sq:
                    sub_adj[i_loc].append(j_loc)
                    sub_adj[j_loc].append(i_loc)

        sub_visited = np.zeros(m, dtype=bool)
        split_parts: List[List[int]] = []
        noise_part: List[int] = []

        for i_loc in range(m):
            if sub_visited[i_loc]:
                continue
            if not sub_adj[i_loc]:
                sub_visited[i_loc] = True
                noise_part.append(int(indices[i_loc]))
                continue
            q = deque([i_loc])
            sub_visited[i_loc] = True
            sub_comp: List[int] = []
            while q:
                u_loc = q.popleft()
                sub_comp.append(int(indices[u_loc]))
                for v_loc in sub_adj[u_loc]:
                    if not sub_visited[v_loc]:
                        sub_visited[v_loc] = True
                        q.append(v_loc)
            split_parts.append(sub_comp)

        valid_splits = [c for c in split_parts if len(c) >= min_size]

        if len(valid_splits) > 0:
            final_clusters.extend(valid_splits)
            discarded.extend(noise_part)
            for c in split_parts:
                if len(c) < min_size:
                    discarded.extend(c)
        else:
            discarded.extend(cluster)

    LOGGER.info(f"Adaptive summary: {len(final_clusters)} clusters final.")
    return ClusterResult(clusters=final_clusters, discarded=discarded)


def refine_clusters_structure(clusters: List[List[int]], feats: np.ndarray, config: Config) -> ClusterResult:
    """
    Post-process clusters: MST splitting, Pruning, and RESCUE of dense cores from garbage.
    Parallelized version.
    """
    try:
        import scipy.spatial.distance as sdist
        from scipy.sparse import csr_matrix, coo_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
    except Exception as exc:  # pragma: no cover
        LOGGER.error(f"Refine requested but SciPy is unavailable: {exc}. Skipping refinement.")
        return ClusterResult(clusters=clusters, discarded=[])

    # --- Настройки ---
    min_size = int(config.clustering.min_size)
    global_threshold = float(config.clustering.threshold)
    
    # MST Split Ratio
    # Если поставить слишком мало (1.5), будет дробить всё подряд.
    # Если слишком много (3.0), пропустит склейки (ваша проблема).
    # Вернем разумный баланс, так как теперь у нас есть Rescue Mode.
    split_ratio = 2.4 
    outlier_sigma = 2.0        
    garbage_density_ratio = 0.6
    
    # Порог спасения: если внутри "мусора" есть точки ближе чем эта дистанция, это ядро.
    # 0.7 от 0.47 = 0.33. Ваша пара (0.15) легко пройдет.
    rescue_threshold_ratio = 0.75
    safe_size = 1 # Используем 1, чтобы фильтрация шла внутри функций, или 5 для скорости

    is_debug = LOGGER.lvlp(3)
    show_progress = LOGGER.lvlp(2)

    def _recursive_mst_split(indices: List[int], sub_feats: np.ndarray) -> List[List[int]]:
        m = len(indices)
        if m < safe_size:
            return [indices]

        dists = sdist.pdist(sub_feats, metric="euclidean")
        if dists.size == 0: return [indices]
        
        d_mat = sdist.squareform(dists)
        tree = minimum_spanning_tree(csr_matrix(d_mat))
        rows, cols = tree.nonzero()
        data = tree.data

        if len(data) == 0: return [indices]

        mean_weight = float(np.mean(data))
        threshold_edge = mean_weight * split_ratio
        max_idx = int(np.argmax(data))
        max_val = float(data[max_idx])

        if max_val > threshold_edge and max_val > 1e-4:
            # Cut logic
            mask = np.ones(len(data), dtype=bool)
            mask[max_idx] = False
            new_graph = coo_matrix((data[mask], (rows[mask], cols[mask])), shape=(m, m))
            n_comps, labels = connected_components(new_graph, directed=False)
            parts = []
            for comp_id in range(n_comps):
                comp_local = np.where(labels == comp_id)[0]
                comp_global = [indices[k] for k in comp_local]
                parts.extend(_recursive_mst_split(comp_global, sub_feats[comp_local]))
            return parts
        return [indices]

    def _prune_outliers(indices: List[int]) -> Tuple[List[int], List[int]]:
        if len(indices) < safe_size:
            return indices, []
        sub_feats = feats[np.array(indices, dtype=int)].astype(np.float32, copy=False)
        centroid = np.mean(sub_feats, axis=0)
        dists = np.linalg.norm(sub_feats - centroid, axis=1)
        mean_d, std_d = float(np.mean(dists)), float(np.std(dists))
        if std_d < 1e-6: return indices, []
        
        limit = mean_d + (outlier_sigma * std_d)
        keep, toss = [], []
        for i, d in enumerate(dists):
            (keep if d <= limit else toss).append(indices[i])
        return keep, toss

    def _rescue_dense_cores(indices: List[int]) -> Tuple[List[List[int]], List[int]]:
        """
        Пытается найти плотные ядра внутри кластера, который был помечен как 'garbage'.
        Использует Connected Components с жестким порогом.
        """
        m = len(indices)
        if m < min_size: 
            return [], indices

        sub_feats = feats[np.array(indices, dtype=int)].astype(np.float32, copy=False)
        
        # Оптимизация памяти: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        # Это избегает создания 3D тензора (N, N, D), который ест память при больших m
        sq_norms = np.sum(sub_feats**2, axis=1)
        # d^2
        d2 = sq_norms[:, None] + sq_norms[None, :] - 2 * (sub_feats @ sub_feats.T)
        # Из-за флоат погрешности могут быть -0.0
        np.maximum(d2, 0.0, out=d2)
        
        # Строгий порог (в квадрате, чтобы не брать корень от матрицы)
        strict_thr_sq = (global_threshold * rescue_threshold_ratio) ** 2
        
        # Строим граф: ребро есть, только если дистанция^2 < strict_thr^2
        adj_matrix = (d2 <= strict_thr_sq).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        
        # Ищем компоненты
        n_comps, labels = connected_components(csr_matrix(adj_matrix), directed=False)
        
        saved_clusters = []
        true_trash = []
        
        for comp_id in range(n_comps):
            comp_local = np.where(labels == comp_id)[0]
            if len(comp_local) >= min_size:
                comp_global = [indices[k] for k in comp_local]
                saved_clusters.append(comp_global)
            else:
                for k in comp_local:
                    true_trash.append(indices[k])
                    
        return saved_clusters, true_trash

    def _process_single_cluster(cluster: List[int]) -> Tuple[List[List[int]], List[int]]:
        """Обработка одного кластера для запуска в потоке."""
        local_final = []
        local_discarded = []

        if len(cluster) < min_size:
            return [], list(cluster)

        # 1. MST Split
        sub_feats = feats[np.array(cluster, dtype=int)].astype(np.float32, copy=False)
        parts = _recursive_mst_split(cluster, sub_feats)
        
        for part in parts:
            if len(part) < min_size:
                local_discarded.extend(part)
                continue

            # 2. Outlier Pruning
            cleaned, outliers = _prune_outliers(part)
            local_discarded.extend(outliers)
            
            if len(cleaned) < min_size:
                local_discarded.extend(cleaned)
                continue

            # 3. Garbage / Density Check
            p_feats = feats[np.array(cleaned, dtype=int)].astype(np.float32, copy=False)
            centroid = np.mean(p_feats, axis=0)
            # Для проверки мусора не обязательно считать pdist, достаточно радиуса от центра
            avg_dist = float(np.mean(np.linalg.norm(p_feats - centroid, axis=1)))
            
            is_garbage = False
            limit_radius = global_threshold * garbage_density_ratio
            threshold_check = limit_radius if len(cleaned) >= max(5, min_size + 1) else (limit_radius * 1.5)

            if global_threshold > 0.0 and avg_dist > threshold_check:
                is_garbage = True

            if is_garbage:
                LOGGER.debug(f"[refine-garbage] Rescue needed (avg_rad={avg_dist:.3f}).")
                
                rescued_cores, hopeless_trash = _rescue_dense_cores(cleaned)
                if rescued_cores:
                    local_final.extend(rescued_cores)
                local_discarded.extend(hopeless_trash)
            else:
                local_final.append(cleaned)
                
        return local_final, local_discarded

    # --- MAIN PARALLEL LOOP ---
    final_clusters: List[List[int]] = []
    discarded: List[int] = []
    
    # Определяем кол-во потоков
    max_workers = os.cpu_count() or 1
    # Если кластеров мало, не создаем лишние потоки
    max_workers = min(max_workers, len(clusters))
    
    if max_workers <= 1:
        # Синхронный режим
        iterator = tqdm(clusters, desc="Refining structure", disable=not show_progress)
        for cl in iterator:
            f, d = _process_single_cluster(cl)
            final_clusters.extend(f)
            discarded.extend(d)
    else:
        # Асинхронный режим
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map сохраняет порядок, но нам он не критичен здесь, 
            # однако для прогресс-бара лучше использовать as_completed или list(tqdm(...))
            results = list(tqdm(
                executor.map(_process_single_cluster, clusters), 
                total=len(clusters), 
                desc="Refining structure (MT)", 
                disable=not show_progress
            ))
            
            for f, d in results:
                final_clusters.extend(f)
                discarded.extend(d)

    LOGGER.info(
        f"Refine summary: {len(final_clusters)} clusters kept, {len(discarded)} items discarded."
    )
    return ClusterResult(clusters=final_clusters, discarded=discarded)

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
    # prepared = _sort_clusters_by_embedding(prepared, feats, config)

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


def _candidate_starts_for_cluster(desired: int, length: int, current_len: int, group_size: int) -> List[int]:
    """
    Generate candidate insertion indices based on group alignment rules.
    
    - If size < group_size: Candidates must fit entirely within a single group. 
      We check the groups that 'desired' touches or is adjacent to.
    - If size >= group_size: We check the range covering the groups the cluster would 
      occupy at 'desired', allowing shifts from edge to edge of those groups.
    """
    # Always include the exact desired position (clamped) as a fallback/baseline
    candidates = {max(0, min(desired, current_len))}
    
    if group_size <= 0:
        return sorted(list(candidates))

    # Identify the range of groups we are interested in
    # We look at the group where 'desired' starts, and where 'desired + length' ends.
    start_g = desired // group_size
    end_g = (desired + length) // group_size
    
    # We broaden the search slightly to ensure we check adjacent group options 
    # if we are near a boundary
    groups_to_check = range(max(0, start_g - 1), end_g + 2)

    for g in groups_to_check:
        g_start = g * group_size
        g_end = g_start + group_size
        
        if length < group_size:
            # Constaint: Cluster must be strictly inside [g_start, g_end]
            # Max start index is g_end - length
            valid_start_min = g_start
            valid_start_max = g_end - length
        else:
            # Constraint: Cluster spans multiple groups.
            # We allow starting anywhere in this group context, effectively sliding 
            # the large cluster through this group frame.
            valid_start_min = g_start
            valid_start_max = g_end # We can start even at the very end of this group
            
        # Validate bounds against logic
        if valid_start_max < valid_start_min:
            continue
            
        # Generate range, clamping to actual sequence length
        for s in range(valid_start_min, valid_start_max + 1):
            if 0 <= s <= current_len:
                candidates.add(s)

    return sorted(list(candidates))

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
    # 1. Base distance cost (neighbors)
    left = seq[start - 1] if start > 0 else None
    right = seq[start] if start < len(seq) else None
    cost = _cluster_neighbor_cost(orient, left, right, feats, use_centroid, centroid)

    # 2. Group alignment penalties/bonuses
    if group_size > 0:
        length = len(orient)
        
        if length > group_size:
            # Logic for Large Clusters
            rem = length % group_size
            
            # Case A: Size is multiple of group_size
            if rem == 0:
                # Bonus for aligning to grid
                if start % group_size == 0:
                    cost -= 0.2
            
            # Case B: Remainder > half group_size
            # Spec: "Just slide left/right with normal priorities" -> No special penalty added
            
            # Case C: Remainder <= half group_size
            # Spec: Prioritize balanced tails (e.g. 5+10+6 > 1+10+10)
                                                                            # elif rem <= group_size / 2:
            # Calculate Left Group Partial (items in the first group)
            # If start is 12 (group 10-19), items are 12..19 -> 8 items.
            offset_in_group = start % group_size
            left_partial = group_size - offset_in_group
            # Verify left_partial isn't larger than total length (shouldn't happen here but safe)
            left_partial = min(left_partial, length)
            
            remaining = length - left_partial
            
            # Calculate Right Group Partial (items in the last group)
            if remaining > 0:
                right_partial = remaining % group_size
                # If modulo is 0, it means it fills the last group completely? 
                # No, rem <= half.
                # Example: Len 21, Start 12 (group 10).
                # Left=8 (12-19). Rem=13. 
                # Next group (20-29) takes 10. Rem=3.
                # Last group takes 3. right_partial=3.
                if right_partial == 0 and remaining >= group_size:
                        # It ended exactly on boundary
                        right_partial = group_size
            else:
                right_partial = 0 # Should not happen if length > group size
            
            # Formula: base + clamp(((L-R)^2)/150 - 0.1, 0.0, 1.0)
            diff = left_partial - right_partial
            penalty = ((diff ** 2) / 150.0) - 0.1
            penalty = max(0.0, min(1.0, penalty))
            cost += penalty

    # 3. Distance from desired position penalty (to avoid drifting too far)
    # Using a small weight to break ties in favor of the intended position
    cost += 0.0005 * abs(start - desired)
    
    return cost


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

def _precompute_base_stats(base_seq: List[int], feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-calculate vectors and gap distances for the base sequence to speed up CLi lookup.
    """
    if not base_seq:
        return np.zeros((0, feats.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    
    # 1. Матрица векторов для base_seq (N_base x Dim)
    base_vecs = feats[np.array(base_seq, dtype=int)].astype(np.float32, copy=False)
    
    # 2. Дистанции между соседями i и i+1 в base_seq (Gap costs)
    # Это те "мостики", которые мы разрываем при вставке.
    # Cost = ||L - C|| + ||R - C|| - ||L - R||
    # Вот этот ||L - R|| мы считаем заранее.
    if len(base_seq) > 1:
        diffs = base_vecs[:-1] - base_vecs[1:]
        gap_dists = np.linalg.norm(diffs, axis=1)
    else:
        gap_dists = np.zeros((0,), dtype=np.float32)
        
    return base_vecs, gap_dists


def _compute_cluster_cli_vectorized(
    items: List[int],
    base_vecs: np.ndarray,
    base_gap_dists: np.ndarray,
    feats: np.ndarray,
    use_centroid: bool,
) -> Tuple[int, List[int], float, np.ndarray]:
    """
    Vectorized version of CLi computation.
    Instead of iterating position by position (Python loop), uses NumPy broadcasting.
    """
    # 1. Определяем "представителя" кластера
    if use_centroid or len(items) == 0:
        centroid = _cluster_embedding(items, feats)
        # Для центроида ориентация не важна, считаем один раз
        # Dists from centroid to ALL base items at once
        # (N_base,)
        dists_to_base = np.linalg.norm(base_vecs - centroid, axis=1)
        
        # Orientations
        c_start_dists = dists_to_base
        c_end_dists = dists_to_base
        centroid_obj = centroid
    else:
        # Edge mode: берем первый и последний элементы
        vec_start = feats[int(items[0])].astype(np.float32, copy=False)
        vec_end = feats[int(items[-1])].astype(np.float32, copy=False)
        
        c_start_dists = np.linalg.norm(base_vecs - vec_start, axis=1)
        c_end_dists = np.linalg.norm(base_vecs - vec_end, axis=1)
        centroid_obj = _cluster_embedding(items, feats) # Просто для возврата

    n_base = len(base_vecs)
    if n_base == 0:
        return 0, items, 0.0, centroid_obj

    # 2. Вычисляем стоимость вставки для всех позиций РАЗОМ
    # Позиций вставки n_base + 1 (от 0 до n_base)
    # Стоимость для внутренних позиций (индексы вставки 1..n-1)
    # base_gap_dists[k] corresponds to gap between k and k+1
    mid_costs_fwd = (
        c_start_dists[:-1] +   # Dist(Base[i-1], Cluster_Start)
        c_end_dists[1:] -      # Dist(Base[i], Cluster_End)
        base_gap_dists         # Dist(Base[i-1], Base[i])
    )
    
    # Стоимость для краев
    # Pos 0: Left=None. Cost = Dist(Base[0], Cluster_Right)
    cost_start_fwd = c_end_dists[0]
    # Pos N: Right=None. Cost = Dist(Base[-1], Cluster_Left)
    cost_end_fwd = c_start_dists[-1]
    
    # --- Reverse Orientation (reversed(items)) ---
    # Cluster is flipped: Start is now items[-1], End is items[0]
    # Left connects to items[-1] (c_end), Right connects to items[0] (c_start)
    
    mid_costs_rev = (
        c_end_dists[:-1] + 
        c_start_dists[1:] - 
        base_gap_dists
    )
    cost_start_rev = c_start_dists[0]
    cost_end_rev = c_end_dists[-1]
    
    # 3. Находим минимум
    # Собираем полные массивы стоимостей
    # [Start, ...Mid..., End]
    
    costs_fwd = np.concatenate(([cost_start_fwd], mid_costs_fwd, [cost_end_fwd]))
    costs_rev = np.concatenate(([cost_start_rev], mid_costs_rev, [cost_end_rev]))
    
    min_idx_fwd = np.argmin(costs_fwd)
    min_val_fwd = costs_fwd[min_idx_fwd]
    
    min_idx_rev = np.argmin(costs_rev)
    min_val_rev = costs_rev[min_idx_rev]
    
    if min_val_fwd <= min_val_rev:
        return int(min_idx_fwd), items, float(min_val_fwd), centroid_obj
    else:
        return int(min_idx_rev), list(reversed(items)), float(min_val_rev), centroid_obj

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
    use_centroid = bool(getattr(config.clustering, "cluster_sort_use_centroid", True))
    show_progress = LOGGER.lvlp(2)

    # Map original cluster id for naming/contiguity
    cluster_id_map: Dict[int, int] = {}
    for cid, cl in enumerate(result.clusters, start=1):
        for idx in cl:
            cluster_id_map[int(idx)] = int(cid)

    # 1. Internal sorting of clusters
    clusters_seq = list(result.clusters)
    cluster_count = len(clusters_seq)

    def _order_single(cluster: Sequence[int]) -> List[int]:
        return _order_cluster_items(list(cluster), feats, config)

    prepared: List[List[int]]
    if cluster_count == 0:
        prepared = []
    else:
        max_workers = max(1, min(cluster_count, os.cpu_count() or 1))
        if max_workers == 1:
            prepared = [_order_single(cl) for cl in clusters_seq]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                prepared = list(executor.map(_order_single, clusters_seq))

    # 2. Internal sorting of discarded (base sequence)
    base_seq = _order_discarded(list(result.discarded), feats, config)
    feats_cache = feats.astype(np.float32, copy=False)

    base_vecs, base_gaps = _precompute_base_stats(base_seq, feats)

    # B. Расчет позиций для каждого кластера
    # Теперь функция _compute_cluster_cli_vectorized выполняется очень быстро,
    # так как внутри неё нет циклов, только матричная алгебра.
    
    def _calc_cli_job_vectorized(args):
        pid, items = args
        best_pos, best_oriented, best_cost, centroid = _compute_cluster_cli_vectorized(
            items, base_vecs, base_gaps, feats, use_centroid
        )
        return {
            "pid": pid,
            "items": best_oriented,
            "best_pos": best_pos,
            "best_cost": best_cost,
            "orig_size": len(items),
            "centroid": centroid,
        }

    job_args = [(pid, list(cl)) for pid, cl in enumerate(prepared)]

    cluster_count = len(job_args)
    max_workers = max(1, min(cluster_count, os.cpu_count() or 1))

    if cluster_count > 0:
        if max_workers <= 1:
            cluster_structs = [_calc_cli_job_vectorized(args) for args in tqdm(job_args, desc="Calc CLi (Vec)", disable=not show_progress)]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                cluster_structs = list(tqdm(
                    executor.map(_calc_cli_job_vectorized, job_args),
                    total=cluster_count,
                    desc="Calc CLi (Vec+MT)",
                    disable=not show_progress
                ))
    else:
        cluster_structs = []

    # Sort clusters by their target insertion index (CLi)
    cluster_structs.sort(key=lambda c: (c["best_pos"], c["best_cost"]))

    # 4. Insertion Phase
    seq: List[int] = list(base_seq)
    
    # We maintain a list of mutable cluster objects because we need to update 
    # their 'best_pos' dynamically as we insert chunks into the sequence.
    # Since we need to pop and shift, using a list as a queue.
    queue = cluster_structs
    
    processed_spans: List[Tuple[int, int, int]] = []

    while queue:
        cl = queue.pop(0)
        
        # The 'best_pos' in cl is now relative to the *current* state of 'seq' 
        # (because we update remaining items after every insertion)
        desired = max(0, min(cl["best_pos"], len(seq)))
        length = len(cl["items"])

        best_start = desired
        best_orient = cl["items"]
        best_cost_place = float("inf")
        
        # Generate candidates: logic handles "fit in group" or "slide in group"
        candidates = _candidate_starts_for_cluster(desired, length, len(seq), group_size)
        
        for start in candidates:
            # Check both orientations
            for orient in (cl["items"], list(reversed(cl["items"]))):
                cost = _placement_cost(seq, start, orient, feats_cache, group_size, desired, use_centroid, cl["centroid"])
                
                # Tie-breaking: lower cost, then closer to desired
                if cost < best_cost_place:
                    best_cost_place = cost
                    best_start = start
                    best_orient = orient
                elif cost == best_cost_place:
                    if abs(start - desired) < abs(best_start - desired):
                        best_start = start
                        best_orient = orient
        
        # Perform Insertion
        # Insert best_orient into seq at best_start
        seq[best_start:best_start] = best_orient
        
        processed_spans.append((cl["pid"], best_start, best_start + length - 1))
        
        LOGGER.debug(
            f"[cluster_sort] pid={cl['pid']} target={desired} -> inserted_at={best_start} "
            f"(cost={best_cost_place:.4f})"
        )

        # 5. Dynamic Update of Subsequent Clusters (CLi Shift)
        # Rule: If we inserted at 'best_start', any target index >= 'best_start' 
        # effectively moves to the right by 'length'.
        # Note: If a cluster target was *before* best_start, it is unaffected.
        for other in queue:
            if other["best_pos"] >= best_start:
                other["best_pos"] += length

    # Logging checks for debug
    for pid, start, end in processed_spans:
        LOGGER.debug(f"[cluster_sort] span pid={pid} start={start} end={end} len={end-start+1}")

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
