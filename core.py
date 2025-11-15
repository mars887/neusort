# core.py

import os
import torch
from database import process_and_cache_features, load_features_from_db
from sorting import sort_images
from search import handle_search_pipeline
import faiss
from cli import LOGGER, ENTERED_GROUPED
from clustering import (
    cluster_by_distance,
    cluster_by_hdbscan,
    cluster_by_dbscan,
    cluster_by_graph,
    cluster_by_mutual_graph,
    apply_pca_whitening,
    export_clusters,
)


def run_sorting_pipeline(config, db_file, index_file):
    """
    Full sorting pipeline: compute/cached features, build FAISS index, sort.
    """
    use_gpu_faiss = (not config.model.use_cpu) and torch.cuda.is_available()

    # 1) Ensure features are computed and cached
    process_and_cache_features(db_file, config)

    # 2) Load features
    feats, paths = load_features_from_db(db_file)
    if not paths or feats is None or len(paths) == 0:
        return

    # 3) Build a FAISS index for sorting helpers
    d = feats.shape[1]
    LOGGER.info(f"Building FAISS index (IndexFlatL2) for {len(paths)} items, dim={d} ...")
    index = faiss.IndexFlatL2(d)
    if use_gpu_faiss:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            LOGGER.info(f"  - Warning: could not move FAISS index to GPU: {e}")
    index.add(feats.astype("float32", copy=False))

    LOGGER.info(f"Proceeding to sorting for {len(paths)} files. This may take a while...")
    sort_images(feats, paths, config)


def run_search_pipeline(config, db_file, index_file):
    """
    Pipeline for the --find mode.
    """
    # Auto-detect query_mode based on what was passed to --find / --query_text
    search_cfg = getattr(config, "search", None)
    if search_cfg is not None:
        find_arg = getattr(search_cfg, "find", None)
        # Skip pipeline mode and respect explicit query_mode from CLI subparams
        if find_arg and find_arg != "__PIPE__" and "find.query_mode" not in ENTERED_GROUPED:
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
            else:
                inferred_mode = "text"
            prev_mode = (getattr(search_cfg, "query_mode", "") or "").lower()
            if prev_mode != inferred_mode:
                LOGGER.info(f"Auto-detected query_mode='{inferred_mode}' for --find input.")
                search_cfg.query_mode = inferred_mode

    feats, paths = load_features_from_db(db_file)
    if not paths or feats is None or len(paths) == 0:
        LOGGER.error("Search aborted: no features found in the database.")
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
        LOGGER.error("Clustering aborted: no features found in the database.")
        return

    # Optional PCA + whitening before clustering
    if getattr(config.clustering, "pca_enabled", False):
        comps = int(getattr(config.clustering, "pca_components", 256))
        whiten = bool(getattr(config.clustering, "pca_whiten", True))
        LOGGER.info(f"Applying PCA preprocessing: components={comps}, whiten={whiten} ...")
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
        LOGGER.info("Running clustering with HDBSCAN (density-based, accuracy-focused)...")
        result = cluster_by_hdbscan(feats, config)
    elif algo == "dbscan":
        LOGGER.info("Running clustering with DBSCAN (density-based) ...")
        result = cluster_by_dbscan(feats, config, use_gpu_faiss)
    elif algo == "cc_graph":
        LOGGER.info("Running clustering with CC ε-graph (connected components) ...")
        result = cluster_by_graph(feats, config, use_gpu_faiss)
    elif algo == "mutual_graph":
        LOGGER.info("Running clustering with Mutual ε-graph (connected components) ...")
        result = cluster_by_mutual_graph(feats, config, use_gpu_faiss)
    else:
        LOGGER.info("Running clustering with distance-based greedy algorithm...")
        result = cluster_by_distance(feats, config, use_gpu_faiss)

    summary_path, cluster_count, discarded_count = export_clusters(paths, feats, result, config)
    LOGGER.info(
        f"Clustering complete: {cluster_count} clusters saved. "
        f"Summary written to: {summary_path}. Discarded items: {discarded_count}."
    )

