import os

TRUE_STRINGS = {'1', 'true', 'yes', 'on'}
FALSE_STRINGS = {'0', 'false', 'no', 'off'}

class Config:
    def __init__(self, args, grouped: dict | None = None):
        grouped = grouped or {}

        def _g(parent: str, key: str, default=None):
            return grouped.get(f"{parent}.{key}", default)

        def _g_bool(parent: str, key: str, default=False):
            val = _g(parent, key, None)
            if val is None:
                return bool(default)
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                lowered = val.strip().lower()
                if lowered in TRUE_STRINGS:
                    return True
                if lowered in FALSE_STRINGS:
                    return False
            return bool(val)
        # Группа: Файлы и пути
        class Files:
            def __init__(self, args):
                self.src_folder = getattr(args, 'input', None) or 'input_images'
                self.dst_folder = getattr(args, 'output', None) or 'sorted_images'
                self.index_file = getattr(args, 'index_file', None) or 'faiss.index'
                self.out_tsv = getattr(args, 'out_tsv', None) or 'neighbor_list.tsv'
        self.files = Files(args)

        # Группа: Модель и извлечение признаков
        class Model:
            def __init__(self, args):
                self.model_name = getattr(args, 'model', None) or 'clip_vit_liaon'
                self.more_scan = bool(getattr(args, 'more_scan', False))
                self.use_cpu = bool(getattr(args, 'use_cpu', False))
                self.batch_size = getattr(args, 'image_batch_size', None) or 1024
                requested_workers = getattr(args, 'feature_workers', None)
                if requested_workers is None or requested_workers <= 0:
                    cpu_count = os.cpu_count() or 1
                    self.feature_workers = max(1, cpu_count)
                else:
                    self.feature_workers = max(1, requested_workers)
                self.gpu_id = 0
        self.model = Model(args)

        # Группа: Сортировка и оптимизация
        class Sorting:
            def __init__(self, args):
                self.lookahead = int(_g('sorting', 'lookahead', 100))
                self.optimizer = str(_g('sorting', 'sort_optimizer', '2opt'))
                self.two_opt_block_size = int(_g('sorting', 'two_opt_block_size', 100))
                self.two_opt_shift = int(_g('sorting', 'two_opt_shift', 90))
                self.neighbors_k_limit = int(_g('sorting', 'neighbors_k_limit', 1024))

                self.strategy = str(_g('sorting', 'sort_strategy', 'farthest_insertion'))

                self.max_cluster_size = 450
                self.bridge_rotate_r  = 32
                self.merge_lookahead  = 10

                self.intra_cluster_refine = True
                self.intra_cluster_refine_threshold = 1.15
                self.intra_cluster_refine_max_swaps = 400

                self.boundary_refine_L = 12

                self.polish_window  = 136
                self.polish_passes  = 3



        self.sorting = Sorting(args)

        class Clustering:
            def __init__(self, args):
                self.enabled = bool(getattr(args, "cluster", False))
                # Algorithm selection for clustering. Supported: 'distance' (existing method), 'hdbscan'.
                algo = str(_g("cluster", "algorithm", "distance")).lower()
                # Accept legacy alias 'graph' by mapping to 'cc_graph'.
                if algo == "graph":
                    algo = "cc_graph"
                if algo not in {"distance", "hdbscan", "dbscan", "cc_graph", "mutual_graph"}:
                    algo = "distance"
                self.algorithm = algo
                threshold = float(_g("cluster", "threshold", 0.35))
                self.threshold = threshold if threshold >= 0.0 else 0.0
                percent = float(_g("cluster", "similarity_percent", 50.0))
                percent = max(0.0, min(100.0, percent))
                self.similarity_percent = percent
                self.similarity_ratio = percent / 100.0
                min_size = int(_g("cluster", "min_size", 2))
                self.min_size = max(1, min_size)
                naming_mode = str(_g("cluster", "naming_mode", "default")).lower()
                if naming_mode not in {"default", "distance", "distance_plus"}:
                    naming_mode = "default"
                self.naming_mode = naming_mode
                raw_save_mode = _g("cluster", "save_mode", "default")
                save_mode = str(raw_save_mode).lower()
                if save_mode not in {"default", "json", "print", "group_filling"}:
                    save_mode = "default"
                self.save_mode = save_mode
                self.save_discarded = _g_bool("cluster", "save_discarded", True)

                # Group-filling parameters (for save_mode=group_filling)
                gsz = int(_g("cluster", "group_filling_size", 10) or 10)
                self.group_filling_size = max(1, gsz)
                split_mode = str(_g("cluster", "splitting_mode", "recluster") or "recluster").lower()
                if split_mode not in {"recluster", "fi"}:  # 'fi' = farthest_insertion_path
                    split_mode = "recluster"
                self.cluster_splitting_mode = split_mode
                # Whether to select fillers by similarity to group centroid
                self.similar_fill = bool(getattr(args, "cluster_similar_fill", False))

                # PCA/whitening pre-processing
                self.pca_enabled = _g_bool("cluster", "pca", False)
                comp = int(getattr(args, "pca_components", 256))
                self.pca_components = max(1, comp)
                raw_pca_whiten = getattr(args, "pca_whiten", "true")
                if isinstance(raw_pca_whiten, bool):
                    pca_whiten = raw_pca_whiten
                elif isinstance(raw_pca_whiten, str):
                    lowered = raw_pca_whiten.strip().lower()
                    if lowered in TRUE_STRINGS:
                        pca_whiten = True
                    elif lowered in FALSE_STRINGS:
                        pca_whiten = False
                    else:
                        pca_whiten = True
                else:
                    pca_whiten = bool(raw_pca_whiten)
                self.pca_whiten = pca_whiten

                # Limits for distance statistics to avoid O(n^2 * d) memory blowups
                self.pairwise_limit = int(getattr(args, "cluster_pairwise_limit", 1200))
                if self.pairwise_limit < 0:
                    self.pairwise_limit = 0
                # Internal chunk size used for streaming distance computations on large clusters
                self.distance_chunk_size = int(getattr(args, "cluster_distance_chunk", 1024))
                if self.distance_chunk_size <= 0:
                    self.distance_chunk_size = 1024
                
        self.clustering = Clustering(args)

        # Группа: Поиск соседей
        class Search:
            def __init__(self, args):
                self.find = getattr(args, 'find', None)
                self.find_neighbors = int(_g('find', 'find_neighbors', 5))
                self.global_knn_batch_size = int(_g('find', 'batch_size', 2048))
                self.tsv_neighbors = int(_g('find', 'tsv_neighbors', 5))
                self.find_result_type = str(_g('find', 'find_result_type', 'both'))
                backend = str(_g('find', 'backend', 'auto') or 'auto').lower()
                if backend not in {'auto', 'regnet', 'clip'}:
                    backend = 'auto'
                self.backend = backend
                mode = str(_g('find', 'query_mode', 'image') or 'image').lower()
                if mode not in {'image', 'text', 'image+text'}:
                    mode = 'image'
                self.query_mode = mode
                fusion = str(_g('find', 'fusion_mode', 'simple') or 'simple').lower()
                if fusion not in {'simple', 'directional'}:
                    fusion = 'simple'
                self.fusion_mode = fusion
                self.image_weight = float(_g('find', 'image_weight', 0.5))
                self.text_weight = float(_g('find', 'text_weight', 0.5))
                self.directional_alpha = float(_g('find', 'directional_alpha', 0.7))
                base_prompt = _g('find', 'base_prompt', 'a photo on a road')
                self.base_prompt = str(base_prompt) if base_prompt is not None else 'a photo on a road'
                self.query_text = getattr(args, 'query_text', None)
        self.search = Search(args)

        # Группа: Логирование и прочее
        class Misc:
            def __init__(self, args):
                self.log_level = getattr(args, 'loglevel', None) or 'default'
                # list_only can be provided only as a sorting subparam
                self.list_only = getattr(args, 'list_only', False) or False
                self.list_objects = getattr(args, "list_objects", False)
                self.move_db = getattr(args, "move_db", None)
                self.cluster = getattr(args, "cluster", False)
        self.misc = Misc(args)
