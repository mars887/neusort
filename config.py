import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

TRUE_STRINGS = {'1', 'true', 'yes', 'on'}
FALSE_STRINGS = {'0', 'false', 'no', 'off'}


def _group_key(parent: str, key: str) -> str:
    return f"{parent}.{key}"


def _group_value(grouped: dict[str, Any], parent: str, key: str, default: Any = None) -> Any:
    return grouped.get(_group_key(parent, key), default)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_STRINGS:
            return True
        if lowered in FALSE_STRINGS:
            return False
    return bool(value)


def _group_bool(grouped: dict[str, Any], parent: str, key: str, default: bool = False) -> bool:
    val = _group_value(grouped, parent, key, None)
    return _coerce_bool(val, default)


def _group_int(grouped: dict[str, Any], parent: str, key: str, default: Optional[int] = None) -> int:
    val = _group_value(grouped, parent, key, None)
    if val is None:
        if default is None:
            raise ValueError(f"Missing integer value for {parent}.{key}")
        return int(default)
    return int(val)


def _group_float(grouped: dict[str, Any], parent: str, key: str, default: Optional[float] = None) -> float:
    val = _group_value(grouped, parent, key, None)
    if val is None:
        if default is None:
            raise ValueError(f"Missing float value for {parent}.{key}")
        return float(default)
    return float(val)


def _group_str(grouped: dict[str, Any], parent: str, key: str, default: Optional[str] = None) -> str:
    val = _group_value(grouped, parent, key, default)
    if val is None:
        return "" if default is None else str(default)
    return str(val)


@dataclass
class FilesConfig:
    src_folder: str
    dst_folder: str
    index_file: str
    out_tsv: str

    @classmethod
    def from_args(cls, args: Any) -> "FilesConfig":
        return cls(
            src_folder=getattr(args, 'input', 'input_images'),
            dst_folder=getattr(args, 'output', 'sorted_images'),
            index_file=getattr(args, 'index_file', 'faiss.index'),
            out_tsv=getattr(args, 'out_tsv', 'neighbor_list.tsv'),
        )


@dataclass
class ModelConfig:
    model_name: str
    more_scan: bool
    use_cpu: bool
    batch_size: int
    feature_workers: int
    gpu_id: int = 0

    @classmethod
    def from_args(cls, args: Any) -> "ModelConfig":
        requested_workers = getattr(args, 'feature_workers', None)
        if requested_workers is None or requested_workers <= 0:
            requested_workers = os.cpu_count() or 1
        batch_size = getattr(args, 'image_batch_size', 1024) or 1024
        return cls(
            model_name=getattr(args, 'model', 'clip_vit_liaon'),
            more_scan=bool(getattr(args, 'more_scan', False)),
            use_cpu=bool(getattr(args, 'use_cpu', False)),
            batch_size=int(batch_size),
            feature_workers=max(1, int(requested_workers)),
        )


@dataclass
class SortingConfig:
    lookahead: int
    optimizer: str
    two_opt_block_size: int
    two_opt_shift: int
    neighbors_k_limit: int
    strategy: str
    max_cluster_size: int = 450
    bridge_rotate_r: int = 32
    merge_lookahead: int = 10
    intra_cluster_refine: bool = True
    intra_cluster_refine_threshold: float = 1.15
    intra_cluster_refine_max_swaps: int = 400
    boundary_refine_L: int = 12
    polish_window: int = 136
    polish_passes: int = 3

    @classmethod
    def from_grouped(cls, grouped: dict[str, Any]) -> "SortingConfig":
        return cls(
            lookahead=_group_int(grouped, 'sorting', 'lookahead', 100),
            optimizer=_group_str(grouped, 'sorting', 'sort_optimizer', '2opt'),
            two_opt_block_size=_group_int(grouped, 'sorting', 'two_opt_block_size', 100),
            two_opt_shift=_group_int(grouped, 'sorting', 'two_opt_shift', 90),
            neighbors_k_limit=_group_int(grouped, 'sorting', 'neighbors_k_limit', 1024),
            strategy=_group_str(grouped, 'sorting', 'sort_strategy', 'farthest_insertion'),
        )


@dataclass
class ClusteringConfig:
    enabled: bool
    algorithm: str
    threshold: float
    similarity_percent: float
    similarity_ratio: float
    min_size: int
    naming_mode: str
    save_mode: str
    save_discarded: bool
    group_filling_size: int
    cluster_splitting_mode: str
    similar_fill: bool
    pca_enabled: bool
    pca_components: int
    pca_whiten: bool
    pairwise_limit: int
    distance_chunk_size: int
    enable_refine: bool

    @classmethod
    def from_args(cls, args: Any, grouped: dict[str, Any]) -> "ClusteringConfig":
        enabled = bool(getattr(args, 'cluster', False))
        algo = _group_str(grouped, 'cluster', 'algorithm', 'distance').lower()
        if algo == 'graph':
            algo = 'cc_graph'
        if algo not in {'distance', 'hdbscan', 'dbscan', 'cc_graph', 'mutual_graph', 'agglomerative', 'agglomerative_complete', 'optics', 'snn', 'rank_mutual', 'adaptive_graph'}:
            algo = 'distance'

        threshold = max(0.0, _group_float(grouped, 'cluster', 'threshold', 0.35))
        percent = _group_float(grouped, 'cluster', 'similarity_percent', 50.0)
        percent = max(0.0, min(100.0, percent))
        min_size = max(1, _group_int(grouped, 'cluster', 'min_size', 2))

        naming_mode = _group_str(grouped, 'cluster', 'naming_mode', 'default').lower()
        if naming_mode not in {'default', 'distance', 'distance_plus'}:
            naming_mode = 'default'

        raw_save_mode = _group_str(grouped, 'cluster', 'save_mode', 'default')
        save_mode = raw_save_mode.lower()
        if save_mode not in {'default', 'json', 'print', 'group_filling', 'cluster_sort'}:
            save_mode = 'default'

        split_mode = _group_str(grouped, 'cluster', 'splitting_mode', 'recluster').lower()
        if split_mode not in {'recluster', 'fi'}:
            split_mode = 'recluster'

        group_filling_size = max(1, _group_int(grouped, 'cluster', 'group_filling_size', 10))

        pca_components = max(1, int(getattr(args, 'pca_components', 256)))
        raw_pca_whiten = getattr(args, 'pca_whiten', 'true')
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

        pairwise_limit = int(getattr(args, 'cluster_pairwise_limit', 1200))
        if pairwise_limit < 0:
            pairwise_limit = 0
        distance_chunk_size = int(getattr(args, 'cluster_distance_chunk', 1024))
        if distance_chunk_size <= 0:
            distance_chunk_size = 1024
        enable_refine = _group_bool(grouped, 'cluster', 'enable_refine', False)

        return cls(
            enabled=enabled,
            algorithm=algo,
            threshold=threshold,
            similarity_percent=percent,
            similarity_ratio=percent / 100.0,
            min_size=min_size,
            naming_mode=naming_mode,
            save_mode=save_mode,
            save_discarded=_group_bool(grouped, 'cluster', 'save_discarded', True),
            group_filling_size=group_filling_size,
            cluster_splitting_mode=split_mode,
            similar_fill=_group_bool(grouped, 'cluster', 'similar_fill', False),
            pca_enabled=_group_bool(grouped, 'cluster', 'pca', False),
            pca_components=pca_components,
            pca_whiten=pca_whiten,
            pairwise_limit=pairwise_limit,
            distance_chunk_size=distance_chunk_size,
            enable_refine=enable_refine,
        )


@dataclass
class SearchConfig:
    find: Optional[str]
    find_neighbors: int
    global_knn_batch_size: int
    tsv_neighbors: int
    find_result_type: str
    backend: str
    query_mode: str
    fusion_mode: str
    image_weight: float
    text_weight: float
    directional_alpha: float
    base_prompt: str
    query_text: Optional[str]

    @classmethod
    def from_args(cls, args: Any, grouped: dict[str, Any]) -> "SearchConfig":
        find = getattr(args, 'find', None)
        backend = _group_str(grouped, 'find', 'backend', 'auto').lower()
        if backend not in {'auto', 'regnet', 'clip'}:
            backend = 'auto'

        query_mode = _group_str(grouped, 'find', 'query_mode', 'image').lower()
        if query_mode not in {'image', 'text', 'image+text'}:
            query_mode = 'image'

        fusion_mode = _group_str(grouped, 'find', 'fusion_mode', 'simple').lower()
        if fusion_mode not in {'simple', 'directional'}:
            fusion_mode = 'simple'

        result_type = _group_str(grouped, 'find', 'find_result_type', 'both').lower()
        if result_type not in {'indexed', 'path', 'both'}:
            result_type = 'both'

        return cls(
            find=find,
            find_neighbors=_group_int(grouped, 'find', 'find_neighbors', 5),
            global_knn_batch_size=_group_int(grouped, 'find', 'batch_size', 2048),
            tsv_neighbors=_group_int(grouped, 'find', 'tsv_neighbors', 5),
            find_result_type=result_type,
            backend=backend,
            query_mode=query_mode,
            fusion_mode=fusion_mode,
            image_weight=_group_float(grouped, 'find', 'image_weight', 0.5),
            text_weight=_group_float(grouped, 'find', 'text_weight', 0.5),
            directional_alpha=_group_float(grouped, 'find', 'directional_alpha', 0.7),
            base_prompt=_group_str(grouped, 'find', 'base_prompt', 'a photo on a road'),
            query_text=getattr(args, 'query_text', None),
        )


@dataclass
class MiscConfig:
    log_level: str
    list_only: bool
    list_objects: bool
    move_db: Optional[tuple[str, str]]
    cluster: bool
    index_only: bool

    @classmethod
    def from_args(cls, args: Any) -> "MiscConfig":
        move_db: Optional[Sequence[str]] = getattr(args, 'move_db', None)
        move_tuple: Optional[tuple[str, str]] = None
        if move_db:
            move_tuple = (move_db[0], move_db[1])
        return cls(
            log_level=getattr(args, 'loglevel', 'default') or 'default',
            list_only=bool(getattr(args, 'list_only', False)),
            list_objects=bool(getattr(args, 'list_objects', False)),
            move_db=move_tuple,
            cluster=bool(getattr(args, 'cluster', False)),
            index_only=bool(getattr(args, 'index_only', False)),
        )


class Config:
    def __init__(self, args: Any, grouped: Optional[dict[str, Any]] = None) -> None:
        grouped = grouped or {}
        self.files = FilesConfig.from_args(args)
        self.model = ModelConfig.from_args(args)
        self.sorting = SortingConfig.from_grouped(grouped)
        self.clustering = ClusteringConfig.from_args(args, grouped)
        self.search = SearchConfig.from_args(args, grouped)
        self.misc = MiscConfig.from_args(args)
