import time
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Callable, Tuple, List, Union
from scipy import stats
from tqdm.auto import tqdm
import warnings


# ====================== CORE EVALUATION METRICS ======================

def path_length(order: np.ndarray, X: np.ndarray) -> float:
    """Calculate the total Euclidean path length of the ordering."""
    if len(order) < 2:
        return 0.0
        
    P = X[order]
    dif = P[1:] - P[:-1]
    seg = np.sqrt(np.sum(dif * dif, axis=1))
    return float(np.sum(seg))


def path_length_normalized(order: np.ndarray, X: np.ndarray) -> float:
    """Normalized path length (divided by number of points)."""
    if len(order) < 2:
        return 0.0
    return path_length(order, X) / (len(order) - 1)


def edge_length_statistics(order: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """Compute statistics of edge lengths in the path."""
    if len(order) < 2:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "max": 0.0, "p90": 0.0, "p99": 0.0}
        
    P = X[order]
    dif = P[1:] - P[:-1]
    seg = np.sqrt(np.sum(dif * dif, axis=1))
    
    return {
        "mean": float(np.mean(seg)),
        "median": float(np.median(seg)),
        "std": float(np.std(seg)),
        "max": float(np.max(seg)),
        "p90": float(np.percentile(seg, 90)),
        "p99": float(np.percentile(seg, 99)),
        "min": float(np.min(seg))
    }


def neighborhood_preservation(
    order: np.ndarray, 
    X: np.ndarray, 
    k: int = 20,
    window_range: List[int] = [10, 25, 50, 100],
    sample: Optional[int] = None,
    seed: int = 42,
    batch_size: int = 1024
) -> Dict[str, float]:
    """
    Evaluate how well the ordering preserves local neighborhoods.
    
    Returns multiple metrics:
    - knn_recall@k_window: Proportion of true k-NN that appear within window
    - continuity_error: Global continuity error (lower is better)
    - local_continuity: Local continuity error for points in the sample
    - rank_correlation: Spearman correlation between rank in original space and path
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    
    if sample is None or sample > N:
        sample = min(5000, N)
        
    chosen = rng.choice(N, size=min(sample, N), replace=False)
    pos = np.empty(N, dtype=np.int32)
    pos[order] = np.arange(N, dtype=np.int32)
    
    # Efficient distance computation with batch processing
    try:
        import faiss
        use_faiss = True
    except ImportError:
        use_faiss = False
    
    # Compute true neighbors with optimized methods
    if use_faiss:
        index = faiss.IndexFlatL2(D)
        index.add(X.astype(np.float32))
        
        # Process in batches to reduce memory usage
        all_neighbors = np.zeros((len(chosen), k), dtype=np.int32)
        all_distances = np.zeros((len(chosen), k), dtype=np.float32)
        
        for i in range(0, len(chosen), batch_size):
            end = min(i + batch_size, len(chosen))
            batch_indices = chosen[i:end]
            _, I = index.search(X[batch_indices].astype(np.float32), k + 1)
            
            # Remove self from neighbors
            for j, idx in enumerate(range(i, end)):
                self_idx = batch_indices[j]
                mask = I[j] != self_idx
                valid_neighbors = I[j][mask][:k]
                all_neighbors[idx] = valid_neighbors
    else:
        # Fallback to numpy implementation with memory optimization
        all_neighbors = np.zeros((len(chosen), k), dtype=np.int32)
        
        for i, qi in enumerate(chosen):
            # Compute distances efficiently
            diff = X - X[qi]
            dists = np.sum(diff ** 2, axis=1)
            dists[qi] = np.inf  # Exclude self
            
            # Get k nearest neighbors
            neighbors = np.argpartition(dists, k)[:k]
            all_neighbors[i] = neighbors[np.argsort(dists[neighbors])]

    # Calculate metrics
    results = {}
    
    # 1. KNN Recall at different windows
    for window in window_range:
        hits = 0
        total = 0
        
        for i, qi in enumerate(chosen):
            p = int(pos[qi])
            left = max(0, p - window)
            right = min(N - 1, p + window)
            window_nodes = set(order[left:right + 1])
            window_nodes.discard(qi)  # Remove self
            
            neighbors = all_neighbors[i]
            hits += sum(1 for n in neighbors if n in window_nodes)
            total += len(neighbors)
            
        results[f"knn_recall@{k}_w{window}"] = hits / max(1, total)
    
    # 2. Global continuity error (GCE)
    # Measures how well the ordering preserves relative distances
    rank_diffs = []
    for i, qi in enumerate(chosen):
        neighbors = all_neighbors[i]
        p_qi = pos[qi]
        
        for n in neighbors:
            p_n = pos[n]
            rank_diff = abs(p_qi - p_n)
            rank_diffs.append(rank_diff)
    
    results["global_continuity_error"] = np.mean(rank_diffs) / N
    
    # 3. Path smoothness - measure of how gradually the path transitions
    if len(order) > 100:
        # Sample points to calculate smoothness
        smoothness_sample = min(1000, len(order) // 10)
        indices = np.linspace(0, len(order) - 2, smoothness_sample, dtype=int)
        
        direction_changes = []
        P = X[order]
        
        for i in indices:
            if i + 2 < len(order):
                v1 = P[i+1] - P[i]
                v2 = P[i+2] - P[i+1]
                # Normalize vectors
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
                # Calculate cosine similarity of directions
                cos_sim = np.dot(v1_norm, v2_norm)
                direction_changes.append(1 - cos_sim)  # 0 = same direction, 2 = opposite
        
        results["path_smoothness"] = 1.0 - np.mean(direction_changes)
    
    return results


def approx_pair_spearman(
    order: np.ndarray, 
    X: np.ndarray, 
    pairs: int = 50000, 
    seed: int = 42
) -> float:
    """
    Spearman correlation between path distance and embedding space distance.
    
    Args:
        order: Path ordering of points
        X: Feature vectors
        pairs: Number of random pairs to sample
        seed: Random seed for reproducibility
    
    Returns:
        Spearman correlation coefficient (-1 to 1, higher is better)
    """
    rng = np.random.default_rng(seed)
    N = len(order)
    
    if N < 2 or pairs <= 0:
        return 0.0
        
    # Limit pairs to reasonable number
    pairs = min(pairs, N * (N - 1) // 2)
    
    # Create position mapping
    pos = np.empty(N, dtype=np.int32)
    pos[order] = np.arange(N, dtype=np.int32)
    
    # Sample unique pairs
    pairs_sampled = 0
    path_dists = []
    space_dists = []
    
    while pairs_sampled < pairs:
        i, j = rng.choice(N, size=2, replace=False)
        if i == j:
            continue
            
        path_dist = abs(pos[i] - pos[j])
        space_dist = np.linalg.norm(X[i] - X[j])
        
        path_dists.append(path_dist)
        space_dists.append(space_dist)
        
        pairs_sampled += 1
    
    # Calculate Spearman correlation
    return stats.spearmanr(path_dists, space_dists)[0]


def boundary_consistency(
    order: np.ndarray,
    cluster_labels: np.ndarray,
    window_size: int = 50
) -> Dict[str, float]:
    """
    Evaluate consistency of cluster boundaries in the ordering.
    
    Args:
        order: Path ordering of points
        cluster_labels: Cluster assignments for each point
        window_size: Size of window to analyze around boundaries
    
    Returns:
        Dictionary with boundary metrics
    """
    if cluster_labels is None:
        return {
            "boundary_consistency": np.nan,
            "avg_boundary_width": np.nan,
            "boundary_sharpness": np.nan
        }
    
    N = len(order)
    if N < window_size * 2:
        return {
            "boundary_consistency": np.nan,
            "avg_boundary_width": np.nan,
            "boundary_sharpness": np.nan
        }
    
    # Find boundaries where cluster changes
    ordered_labels = cluster_labels[order]
    boundaries = np.where(ordered_labels[:-1] != ordered_labels[1:])[0]
    
    if len(boundaries) == 0:
        return {
            "boundary_consistency": 1.0,  # Perfect consistency if no boundaries
            "avg_boundary_width": 0.0,
            "boundary_sharpness": 1.0
        }
    
    # Analyze each boundary
    boundary_widths = []
    boundary_sharpness = []
    
    for b in boundaries:
        start = max(0, b - window_size // 2)
        end = min(N, b + window_size // 2 + 1)
        
        window_labels = ordered_labels[start:end]
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        
        # Boundary width: how many points around boundary have mixed labels
        dominant_label_before = ordered_labels[max(0, b - 10):b+1]
        dominant_label_after = ordered_labels[b+1:min(N, b + 11)]
        
        label_before = stats.mode(dominant_label_before, keepdims=False).mode
        label_after = stats.mode(dominant_label_after, keepdims=False).mode
        
        transition_start = b
        while transition_start > start and ordered_labels[transition_start] == label_before:
            transition_start -= 1
            
        transition_end = b + 1
        while transition_end < end and ordered_labels[transition_end] == label_after:
            transition_end += 1
            
        boundary_width = transition_end - transition_start
        boundary_widths.append(boundary_width)
        
        # Boundary sharpness: how abruptly the transition happens
        # 1.0 = perfect sharp boundary, 0.0 = gradual transition
        if len(counts) >= 2:
            sorted_counts = np.sort(counts)[::-1]
            dominant_ratio = sorted_counts[0] / (sorted_counts[0] + sorted_counts[1])
            sharpness = dominant_ratio
        else:
            sharpness = 1.0
            
        boundary_sharpness.append(sharpness)
    
    # Overall metrics
    avg_boundary_width = np.mean(boundary_widths)
    avg_sharpness = np.mean(boundary_sharpness)
    
    # Boundary consistency: how consistently boundaries separate the same clusters
    # Count transitions between specific cluster pairs
    transition_pairs = {}
    for b in boundaries:
        if b > 0 and b < N - 1:
            label_before = ordered_labels[b]
            label_after = ordered_labels[b + 1]
            pair = tuple(sorted([label_before, label_after]))
            transition_pairs[pair] = transition_pairs.get(pair, 0) + 1
    
    if transition_pairs:
        # Higher consistency if fewer unique transition types
        num_transition_types = len(transition_pairs)
        total_transitions = sum(transition_pairs.values())
        # Normalize: 1.0 means all boundaries are between the same cluster pair
        consistency = 1.0 - (num_transition_types - 1) / max(1, total_transitions - 1)
    else:
        consistency = 1.0
    
    return {
        "boundary_consistency": float(consistency),
        "avg_boundary_width": float(avg_boundary_width),
        "boundary_sharpness": float(avg_sharpness),
        "num_boundaries": len(boundaries)
    }


def label_run_length(
    order: np.ndarray, 
    labels: Optional[np.ndarray] = None,
    min_cluster_size: int = 5
) -> Dict[str, float]:
    """
    Enhanced analysis of label run lengths in the ordering.
    
    Returns multiple statistics about how labels are grouped.
    """
    if labels is None:
        return {
            "label_runlen_mean": np.nan,
            "label_runlen_median": np.nan,
            "label_homogeneity": np.nan,
            "num_label_transitions": np.nan
        }
    
    labels = np.asarray(labels)
    if len(order) == 0:
        return {
            "label_runlen_mean": 0,
            "label_runlen_median": 0,
            "label_homogeneity": 0,
            "num_label_transitions": 0
        }
    
    # Filter out small clusters if needed
    if min_cluster_size > 1:
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[counts >= min_cluster_size]
        mask = np.isin(labels, valid_labels)
        filtered_order = order[mask[order]]
        filtered_labels = labels[mask]
    else:
        filtered_order = order
        filtered_labels = labels
    
    if len(filtered_order) == 0:
        return {
            "label_runlen_mean": 0,
            "label_runlen_median": 0,
            "label_homogeneity": 0,
            "num_label_transitions": 0
        }
    
    # Calculate run lengths
    runs = []
    current_label = filtered_labels[filtered_order[0]]
    current_run = 1
    
    label_transitions = 0
    
    for i in range(1, len(filtered_order)):
        next_label = filtered_labels[filtered_order[i]]
        if next_label == current_label:
            current_run += 1
        else:
            runs.append(current_run)
            current_label = next_label
            current_run = 1
            label_transitions += 1
    
    runs.append(current_run)
    
    # Calculate homogeneity score (higher is better)
    # Ratio of within-run pairs that have the same label vs all possible same-label pairs
    total_same_label_pairs = 0
    within_run_same_label_pairs = 0
    
    unique_labels = np.unique(filtered_labels)
    
    for label in unique_labels:
        label_mask = (filtered_labels == label)
        n_label = np.sum(label_mask)
        
        if n_label > 1:
            total_same_label_pairs += n_label * (n_label - 1) // 2
            
            # Count within-run pairs
            run_start = 0
            for i in range(1, len(filtered_order)):
                if filtered_labels[filtered_order[i]] != filtered_labels[filtered_order[i-1]]:
                    # End of a run
                    run_labels = filtered_labels[filtered_order[run_start:i]]
                    n_in_run = np.sum(run_labels == label)
                    if n_in_run > 1:
                        within_run_same_label_pairs += n_in_run * (n_in_run - 1) // 2
                    run_start = i
            
            # Last run
            run_labels = filtered_labels[filtered_order[run_start:]]
            n_in_run = np.sum(run_labels == label)
            if n_in_run > 1:
                within_run_same_label_pairs += n_in_run * (n_in_run - 1) // 2
    
    homogeneity = within_run_same_label_pairs / max(1, total_same_label_pairs)
    
    return {
        "label_runlen_mean": float(np.mean(runs)),
        "label_runlen_median": float(np.median(runs)),
        "label_homogeneity": float(homogeneity),
        "num_label_transitions": float(label_transitions),
        "num_runs": float(len(runs))
    }


def sequential_mAP_at_k(
    order: np.ndarray,
    labels: Optional[np.ndarray] = None,
    K: int = 50,
    sample: int = 2000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Enhanced sequential mAP calculation with multiple K values.
    
    Returns metrics for different window sizes to show retrieval quality
    at different scales.
    """
    if labels is None:
        return {
            "sequential_mAP@10": np.nan,
            "sequential_mAP@50": np.nan,
            "sequential_mAP@100": np.nan
        }
    
    rng = np.random.default_rng(seed)
    N = len(order)
    if N == 0:
        return {
            "sequential_mAP@10": 0.0,
            "sequential_mAP@50": 0.0,
            "sequential_mAP@100": 0.0
        }
    
    sample = min(sample, N)
    chosen = rng.choice(N, size=sample, replace=False)
    pos = np.empty(N, dtype=np.int32)
    pos[order] = np.arange(N, dtype=np.int32)
    
    results = {}
    window_sizes = [10, 50, 100]
    
    for window in window_sizes:
        aps = []
        
        for qi in chosen:
            y = labels[qi]
            p = int(pos[qi])
            left = max(0, p - window // 2)
            right = min(N, p + window // 2 + 1)
            
            cand = order[left:right]
            # Exclude the query itself
            cand = cand[cand != qi]
            
            if len(cand) == 0:
                continue
                
            rel = (labels[cand] == y).astype(np.int32)
            
            if np.sum(rel) == 0:
                aps.append(0.0)
                continue
                
            # Calculate precision at each relevant position
            positions = np.where(rel == 1)[0] + 1  # +1 for 1-based indexing
            precisions = np.zeros(len(positions))
            
            for i, pos_idx in enumerate(positions):
                precisions[i] = np.sum(rel[:pos_idx]) / pos_idx
                
            ap = np.mean(precisions)
            aps.append(ap)
        
        results[f"sequential_mAP@{window}"] = float(np.mean(aps)) if aps else 0.0
    
    return results


def curvature_analysis(
    order: np.ndarray,
    X: np.ndarray,
    window_size: int = 10
) -> Dict[str, float]:
    """
    Analyze the curvature and smoothness of the path in feature space.
    
    Higher values indicate smoother paths with more consistent direction changes.
    """
    if len(order) < window_size + 2:
        return {
            "curvature_mean": 0.0,
            "curvature_std": 0.0,
            "direction_consistency": 0.0
        }
    
    P = X[order]
    n = len(P)
    
    curvatures = []
    direction_changes = []
    
    # Calculate curvature using triplets of points
    for i in range(1, n - 1):
        p_prev = P[i-1]
        p_curr = P[i]
        p_next = P[i+1]
        
        # Vectors
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            continue
            
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2
        
        # Angle between vectors (0 to π)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Curvature (higher = sharper turn)
        curvature = angle / (norm1 + norm2)
        curvatures.append(curvature)
        
        # Direction consistency (1 = same direction, -1 = opposite)
        direction_changes.append(cos_angle)
    
    if not curvatures:
        return {
            "curvature_mean": 0.0,
            "curvature_std": 0.0,
            "direction_consistency": 0.0
        }
    
    # Calculate metrics
    curvature_mean = np.mean(curvatures)
    curvature_std = np.std(curvatures)
    direction_consistency = np.mean(direction_changes)
    
    return {
        "curvature_mean": float(curvature_mean),
        "curvature_std": float(curvature_std),
        "direction_consistency": float(direction_consistency)
    }


# ====================== VISUALIZATION HELPERS ======================

def visualize_path_quality(
    X: np.ndarray,
    order: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Path Quality Visualization",
    figsize: Tuple[int, int] = (15, 10),
    output_file: Optional[str] = None
):
    """
    Create comprehensive visualization of path quality metrics.
    
    Args:
        X: Feature vectors (supports 2D or 3D for direct visualization,
            or higher dimensions for PCA projection)
        order: Path ordering
        labels: Optional cluster labels for coloring
        title: Plot title
        figsize: Figure size
        output_file: If provided, save figure to this path
    """
    # Check if we can visualize directly or need dimensionality reduction
    D = X.shape[1]
    needs_reduction = D > 3
    
    if needs_reduction:
        # Use PCA to reduce to 2D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        subtitle = f"PCA projection ({explained_variance[0]:.1%}, {explained_variance[1]:.1%} explained variance)"
    else:
        X_vis = X[:, :2] if D > 2 else X
        subtitle = f"Direct visualization in {D}D space"
    
    # Calculate edge lengths for coloring
    P = X_vis[order]
    edge_lengths = np.linalg.norm(P[1:] - P[:-1], axis=1)
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(pad=4.0)
    
    # 1. Path visualization
    ax = axs[0, 0]
    
    # Plot path
    for i in range(len(P) - 1):
        color = plt.cm.viridis(min(1.0, edge_lengths[i] / np.percentile(edge_lengths, 90)))
        ax.plot(P[i:i+2, 0], P[i:i+2, 1], c=color, linewidth=1, alpha=0.7)
    
    # Plot points
    if labels is not None:
        scatter = ax.scatter(P[:, 0], P[:, 1], c=labels[order], cmap='tab20', s=15, alpha=0.8)
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
    else:
        ax.scatter(P[:, 0], P[:, 1], c=np.arange(len(P)), cmap='viridis', s=15, alpha=0.8)
    
    # Mark start and end points
    ax.scatter(P[0, 0], P[0, 1], c='green', s=100, marker='o', edgecolors='black', label='Start')
    ax.scatter(P[-1, 0], P[-1, 1], c='red', s=100, marker='s', edgecolors='black', label='End')
    ax.legend()
    
    ax.set_title('Path Visualization')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)
    
    # 2. Edge length distribution
    ax = axs[0, 1]
    ax.hist(edge_lengths, bins=50, alpha=0.7, color='blue')
    ax.axvline(np.mean(edge_lengths), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(edge_lengths):.3f}')
    ax.axvline(np.percentile(edge_lengths, 90), color='green', linestyle='dashed', linewidth=2, label=f'90th percentile: {np.percentile(edge_lengths, 90):.3f}')
    ax.set_title('Edge Length Distribution')
    ax.set_xlabel('Edge Length')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Neighborhood preservation
    window_sizes = [10, 25, 50, 100]
    recall_values = []
    
    pos = np.empty(len(order), dtype=np.int32)
    pos[order] = np.arange(len(order), dtype=np.int32)
    
    # Calculate k-NN for a sample of points
    sample_size = min(1000, len(order))
    sample_indices = np.random.choice(len(order), size=sample_size, replace=False)
    
    # Use approximate nearest neighbors for efficiency
    try:
        import faiss
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X.astype(np.float32))
        k = 20
        _, neighbors = index.search(X[sample_indices].astype(np.float32), k + 1)
        neighbors = neighbors[:, 1:]  # Remove self
    except ImportError:
        # Fallback to brute force for small datasets
        neighbors = np.zeros((sample_size, 20), dtype=np.int32)
        for i, idx in enumerate(sample_indices):
            dists = np.linalg.norm(X - X[idx], axis=1)
            dists[idx] = np.inf
            neighbors[i] = np.argsort(dists)[:20]
    
    for window in window_sizes:
        hits = 0
        total = 0
        for i, idx in enumerate(sample_indices):
            p = pos[idx]
            window_start = max(0, p - window)
            window_end = min(len(order), p + window + 1)
            window_nodes = set(order[window_start:window_end])
            window_nodes.discard(idx)  # Remove self
            
            for neighbor in neighbors[i]:
                total += 1
                if neighbor in window_nodes:
                    hits += 1
        
        recall_values.append(hits / max(1, total))
    
    ax = axs[1, 0]
    ax.plot(window_sizes, recall_values, 'o-', linewidth=2, markersize=8)
    ax.set_title('Neighborhood Preservation')
    ax.set_xlabel('Window Size')
    ax.set_ylabel('Recall@20')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # 4. Path position vs. feature space distance
    sample_size = min(500, len(order))
    sample_indices = np.random.choice(len(order), size=sample_size, replace=False)
    
    distances = []
    position_diffs = []
    
    for i in range(len(sample_indices)):
        for j in range(i + 1, len(sample_indices)):
            idx1 = sample_indices[i]
            idx2 = sample_indices[j]
            
            dist = np.linalg.norm(X[idx1] - X[idx2])
            pos_diff = abs(pos[idx1] - pos[idx2])
            
            distances.append(dist)
            position_diffs.append(pos_diff)
    
    ax = axs[1, 1]
    ax.scatter(distances, position_diffs, alpha=0.5, s=10)
    ax.set_title('Position Difference vs. Feature Distance')
    ax.set_xlabel('Feature Space Distance')
    ax.set_ylabel('Position Difference')
    ax.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    
    plt.show()


def compare_algorithms(
    X: np.ndarray,
    algorithms: Dict[str, Callable[[], np.ndarray]],
    labels: Optional[np.ndarray] = None,
    visualize: bool = True,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple path-finding algorithms across comprehensive metrics.
    
    Args:
        X: Feature vectors
        algorithms: Dictionary mapping algorithm names to functions that generate orderings
        labels: Optional cluster labels for label-based metrics
        visualize: Whether to create comparison visualizations
        verbose: Whether to print progress information
    
    Returns:
        Dictionary of metrics for each algorithm
    """
    results = {}
    run_times = {}
    
    # Run each algorithm and compute metrics
    for name, algorithm in algorithms.items():
        if verbose:
            print(f"Running {name}...")
        
        start_time = time.time()
        order = algorithm()
        run_time = time.time() - start_time
        run_times[name] = run_time
        
        # Compute all metrics
        metrics = evaluate_order_metrics(order, X, labels)
        metrics["runtime_seconds"] = run_time
        results[name] = metrics
        
        if verbose:
            print(f"  Completed in {run_time:.2f} seconds")
            print("  Metrics:")
            for metric, value in metrics.items():
                if not isinstance(value, dict) and not np.isnan(value):
                    print(f"    {metric}: {value:.4f}")
    
    # Create comparison visualizations if requested
    if visualize and len(algorithms) > 1:
        # Bar chart comparing key metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Select key metrics to compare
        key_metrics = [
            "path_length_normalized",
            "neighborhood_preservation_knn_recall@20_w50",
            "curvature_direction_consistency"
        ]
        
        # Prepare data
        names = list(results.keys())
        metric_values = {metric: [] for metric in key_metrics}
        
        for name in names:
            for metric in key_metrics:
                value = results[name].get(metric, np.nan)
                metric_values[metric].append(value if not np.isnan(value) else 0)
        
        # Plot 1: Key metrics comparison
        x = np.arange(len(names))
        width = 0.25
        
        for i, metric in enumerate(key_metrics):
            ax1.bar(x + i * width, metric_values[metric], width, label=metric)
        
        ax1.set_title('Algorithm Comparison: Key Metrics')
        ax1.set_xticks(x + width * (len(key_metrics) - 1) / 2)
        ax1.set_xticklabels(names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Runtime comparison
        runtimes = [run_times[name] for name in names]
        ax2.bar(names, runtimes)
        ax2.set_title('Algorithm Runtime Comparison')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results


# ====================== MAIN EVALUATION FUNCTION ======================

def evaluate_order_metrics(
    order: np.ndarray, 
    X: np.ndarray, 
    labels: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Comprehensive evaluation of path ordering quality.
    
    This function computes a wide range of metrics that evaluate different
    aspects of path quality:
    
    1. Path geometry (length, smoothness, curvature)
    2. Neighborhood preservation (how well local structure is maintained)
    3. Label consistency (if labels are provided)
    4. Global structure preservation
    
    Args:
        order: The ordering/path to evaluate (array of indices)
        X: Feature vectors (N, D)
        labels: Optional cluster labels for each point (N,)
        verbose: Whether to print progress information
    
    Returns:
        Dictionary containing all computed metrics
    """
    if verbose:
        print("Evaluating path quality metrics...")
    
    # Validate inputs
    n = len(order)
    if n == 0:
        warnings.warn("Empty ordering provided. All metrics will be zero or NaN.")
        return {metric: 0.0 for metric in [
            "path_length", "path_length_normalized", "spearman_pair_corr"
        ]}
    
    if n != X.shape[0]:
        raise ValueError(f"Mismatch between ordering length ({n}) and feature matrix size ({X.shape[0]})")
    
    # Basic metrics
    metrics = {}
    
    # Path length metrics
    metrics["path_length"] = path_length(order, X)
    metrics["path_length_normalized"] = path_length_normalized(order, X)
    
    # Edge statistics
    edge_stats = edge_length_statistics(order, X)
    for stat_name, value in edge_stats.items():
        metrics[f"edge_{stat_name}"] = value
    
    # Neighborhood preservation metrics
    if verbose:
        print("  Computing neighborhood preservation...")
    
    neighborhood_metrics = neighborhood_preservation(order, X, k=20, sample=min(5000, n))
    for metric_name, value in neighborhood_metrics.items():
        metrics[f"neighborhood_preservation_{metric_name}"] = value
    
    # Path smoothness and curvature
    if verbose:
        print("  Computing path smoothness and curvature...")
    
    curvature_metrics = curvature_analysis(order, X, window_size=10)
    for metric_name, value in curvature_metrics.items():
        metrics[f"curvature_{metric_name}"] = value
    
    # Spearman correlation
    metrics["spearman_pair_corr"] = approx_pair_spearman(order, X, pairs=min(50000, n * (n - 1) // 2))
    
    # Label-based metrics (if labels provided)
    if labels is not None and len(labels) == n:
        if verbose:
            print("  Computing label-based metrics...")
        
        # Label run length analysis
        run_length_metrics = label_run_length(order, labels, min_cluster_size=5)
        for metric_name, value in run_length_metrics.items():
            metrics[metric_name] = value
        
        # Sequential mAP
        map_metrics = sequential_mAP_at_k(order, labels, sample=min(2000, n))
        for metric_name, value in map_metrics.items():
            metrics[metric_name] = value
        
        # Boundary consistency
        boundary_metrics = boundary_consistency(order, labels, window_size=50)
        for metric_name, value in boundary_metrics.items():
            metrics[f"boundary_{metric_name}"] = value
    
    if verbose:
        print("Evaluation complete.")
    
    return metrics