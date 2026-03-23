# core.py

import os
import torch
import runtime_state as runtime
from database import process_and_cache_features, load_features_from_db
from sorting import sort_images
from search import handle_search_pipeline
from clustering import (
    cluster_by_distance,
    cluster_by_hdbscan,
    cluster_by_agglomerative,
    cluster_by_agglomerative_complete,
    cluster_by_optics,
    cluster_by_snn,
    cluster_by_rank_mutual,
    cluster_by_adaptive_graph,
    cluster_by_dbscan,
    cluster_by_graph,
    cluster_by_mutual_graph,
    refine_clusters_structure,
    apply_pca_whitening,
    export_clusters,
)


IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def _looks_like_image_path(raw: str) -> bool:
    if not raw:
        return False
    normalized = os.path.normpath(raw)
    if os.path.isabs(normalized):
        return True
    if any(sep in raw for sep in (os.sep, "/", "\\")):
        return True
    _, ext = os.path.splitext(normalized)
    return ext.lower() in IMAGE_FILE_EXTENSIONS


def run_sorting_pipeline(config, db_file, index_file):
    """
    Full sorting pipeline: compute/cache features and sort.
    """
    # 1) Ensure features are computed and cached
    process_and_cache_features(db_file, config)

    # 2) Load features
    feats, paths = load_features_from_db(db_file)
    if not paths or feats is None or len(paths) == 0:
        return

    runtime.LOGGER.info(f"Proceeding to sorting for {len(paths)} files. This may take a while...")
    sort_images(feats, paths, config)


def run_search_pipeline(config, db_file, index_file, entered_grouped=None):
    """
    Pipeline for the --find mode.
    """
    # Auto-detect query_mode based on what was passed to --find / --query_text
    search_cfg = getattr(config, "search", None)
    entered_grouped = entered_grouped or {}
    if search_cfg is not None:
        find_arg = getattr(search_cfg, "find", None)
        # Skip pipeline mode and respect explicit query_mode from CLI subparams
        if find_arg and find_arg != "__PIPE__" and "find.query_mode" not in entered_grouped:
            raw = str(find_arg).strip()
            # Strip surrounding quotes if the user passed a quoted path literally
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
                raw = raw[1:-1].strip()
            normalized = os.path.normpath(raw) if raw else raw
            has_text = bool(getattr(search_cfg, "query_text", None))
            is_file = bool(normalized) and os.path.isfile(normalized)
            if is_file:
                # Normalize stored path so downstream code works even if slashes/quotes differ
                search_cfg.find = normalized
                inferred_mode = "image+text" if has_text else "image"
            elif _looks_like_image_path(raw):
                inferred_mode = "image+text" if has_text else "image"
            else:
                inferred_mode = "text"
            prev_mode = (getattr(search_cfg, "query_mode", "") or "").lower()
            if prev_mode != inferred_mode:
                runtime.LOGGER.info(f"Auto-detected query_mode='{inferred_mode}' for --find input.")
                search_cfg.query_mode = inferred_mode

    feats, paths = load_features_from_db(db_file)
    if not paths or feats is None or len(paths) == 0:
        runtime.LOGGER.error("Search aborted: no features found in the database.")
        return

    handle_search_pipeline(config, feats, paths)


def run_clustering_pipeline(config, db_file, index_file):
    """
    Pipeline for the --cluster mode.
    """
    use_gpu_faiss = (not config.model.use_cpu) and torch.cuda.is_available()

    process_and_cache_features(db_file, config)

    feats, paths = load_features_from_db(db_file)
    if not paths or feats is None or len(paths) == 0:
        runtime.LOGGER.error("Clustering aborted: no features found in the database.")
        return

    # Optional PCA + whitening before clustering
    if getattr(config.clustering, "pca_enabled", False):
        comps = int(getattr(config.clustering, "pca_components", 256))
        whiten = bool(getattr(config.clustering, "pca_whiten", True))
        runtime.LOGGER.info(f"Applying PCA preprocessing: components={comps}, whiten={whiten} ...")
        feats, pca_meta = apply_pca_whitening(
            feats, n_components=comps, whiten=whiten, log=(config.misc.log_level == "default")
        )
        try:
            setattr(config.clustering, "_pca_effective_components", int(pca_meta.get("components", comps)))
        except Exception:
            pass

    # Choose clustering algorithm based on configuration
    algo = getattr(config.clustering, "algorithm", "distance")
    if algo == "graph":  # legacy alias
        algo = "cc_graph"

    if algo == "hdbscan":
        runtime.LOGGER.info("Running clustering with HDBSCAN (density-based, accuracy-focused)...")
        result = cluster_by_hdbscan(feats, config)
    elif algo == "agglomerative":
        runtime.LOGGER.info("Running clustering with Agglomerative Hierarchical (Ward) ...")
        result = cluster_by_agglomerative(feats, config)
    elif algo == "agglomerative_complete":
        runtime.LOGGER.info("Running clustering with Agglomerative (Complete Linkage) ...")
        result = cluster_by_agglomerative_complete(feats, config)
    elif algo == "optics":
        runtime.LOGGER.info("Running clustering with OPTICS (xi method) ...")
        result = cluster_by_optics(feats, config)
    elif algo == "dbscan":
        runtime.LOGGER.info("Running clustering with DBSCAN (density-based) ...")
        result = cluster_by_dbscan(feats, config, use_gpu_faiss)
    elif algo == "cc_graph":
        runtime.LOGGER.info("Running clustering with CC ε-graph (connected components) ...")
        result = cluster_by_graph(feats, config, use_gpu_faiss)
    elif algo == "mutual_graph":
        runtime.LOGGER.info("Running clustering with Mutual ε-graph (connected components) ...")
        result = cluster_by_mutual_graph(feats, config, use_gpu_faiss)
    elif algo == "snn":
        runtime.LOGGER.info("Running clustering with Shared Nearest Neighbors (SNN) ...")
        result = cluster_by_snn(feats, config, use_gpu_faiss)
    elif algo == "rank_mutual":
        runtime.LOGGER.info("Running clustering with Rank-Based Reciprocal k-NN ...")
        result = cluster_by_rank_mutual(feats, config, use_gpu_faiss)
    elif algo == "adaptive_graph":
        runtime.LOGGER.info("Running clustering with Adaptive Mutual Graph ...")
        result = cluster_by_adaptive_graph(feats, config, use_gpu_faiss)
    else:
        runtime.LOGGER.info("Running clustering with distance-based greedy algorithm...")
        result = cluster_by_distance(feats, config, use_gpu_faiss)

    if getattr(config.clustering, "enable_refine", False):
        runtime.LOGGER.info("Refining clusters (split/prune/garbage filter)...")
        refined = refine_clusters_structure(result.clusters, feats, config)
        refined.discarded.extend(result.discarded)
        result = refined

    summary_path, cluster_count, discarded_count = export_clusters(paths, feats, result, config)
    runtime.LOGGER.info(
        f"Clustering complete: {cluster_count} clusters saved. "
        f"Summary written to: {summary_path}. Discarded items: {discarded_count}."
    )

