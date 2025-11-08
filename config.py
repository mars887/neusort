import os

TRUE_STRINGS = {'1', 'true', 'yes', 'on'}
FALSE_STRINGS = {'0', 'false', 'no', 'off'}

class Config:
    def __init__(self, args):
        # Группа: Файлы и пути
        class Files:
            def __init__(self, args):
                self.src_folder = args.input
                self.dst_folder = args.output
                self.index_file = args.index_file
                self.out_tsv = args.out_tsv
        self.files = Files(args)

        # Группа: Модель и извлечение признаков
        class Model:
            def __init__(self, args):
                self.model_name = args.model
                self.more_scan = args.more_scan
                self.use_cpu = args.use_cpu
                self.batch_size = args.image_batch_size  # Размер батча для изображений
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
                self.lookahead = args.lookahead
                self.optimizer = args.sort_optimizer
                self.two_opt_block_size = args.two_opt_block_size
                self.two_opt_shift = args.two_opt_shift
                self.neighbors_k_limit = args.neighbors_k_limit
                
                self.strategy = args.sort_strategy

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
                algo = str(getattr(args, "cluster_algorithm", "distance")).lower()
                # Accept legacy alias 'graph' by mapping to 'cc_graph'.
                if algo == "graph":
                    algo = "cc_graph"
                if algo not in {"distance", "hdbscan", "dbscan", "cc_graph", "mutual_graph"}:
                    algo = "distance"
                self.algorithm = algo
                threshold = float(getattr(args, "threshold", 0.35))
                self.threshold = threshold if threshold >= 0.0 else 0.0
                percent = float(getattr(args, "similarity_percent", 50.0))
                percent = max(0.0, min(100.0, percent))
                self.similarity_percent = percent
                self.similarity_ratio = percent / 100.0
                min_size = int(getattr(args, "cluster_min_size", 2))
                self.min_size = max(1, min_size)
                naming_mode = str(getattr(args, "cluster_naming_mode", "default")).lower()
                if naming_mode not in {"default", "distance", "distance_plus"}:
                    naming_mode = "default"
                self.naming_mode = naming_mode
                raw_save_mode = getattr(args, "save_mode", "default")
                save_mode = str(raw_save_mode).lower()
                if save_mode not in {"default", "json", "print"}:
                    save_mode = "default"
                self.save_mode = save_mode
                raw_discarded = getattr(args, "save_discarded", "true")
                if isinstance(raw_discarded, bool):
                    save_discarded = raw_discarded
                elif isinstance(raw_discarded, str):
                    lowered = raw_discarded.strip().lower()
                    if lowered in TRUE_STRINGS:
                        save_discarded = True
                    elif lowered in FALSE_STRINGS:
                        save_discarded = False
                    else:
                        save_discarded = True
                else:
                    save_discarded = bool(raw_discarded)
                self.save_discarded = save_discarded

                # PCA/whitening pre-processing
                self.pca_enabled = bool(getattr(args, "cluster_pca", False))
                comp = int(getattr(args, "cluster_pca_components", 256))
                self.pca_components = max(1, comp)
                raw_pca_whiten = getattr(args, "cluster_pca_whiten", "true")
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
                self.find = args.find
                self.find_neighbors = args.find_neighbors
                self.global_knn_batch_size = args.batch_size  # Размер батча для глобального k-NN
                self.tsv_neighbors = args.tsv_neighbors
                self.find_result_type = args.find_result_type
        self.search = Search(args)

        # Группа: Логирование и прочее
        class Misc:
            def __init__(self, args):
                self.log_level = args.loglevel
                self.list_only = args.list_only
                self.list_objects = getattr(args, "list_objects", False)
                self.move_db = getattr(args, "move_db", None)
                self.cluster = getattr(args, "cluster", False)
        self.misc = Misc(args)
