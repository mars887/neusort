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
        self.misc = Misc(args)