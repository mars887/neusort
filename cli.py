# ---------------------------------------------------------------------------- #
#                              Аргументы и Конфигурация                        #
# ---------------------------------------------------------------------------- #
import argparse
import re
import sys
import textwrap
from typing import List, Tuple

import torch
from config import Config
from logger import CustomLogger, LogLevel

# --- Централизованная конфигурация аргументов ---
PROGRAMM_ARGS = {
    "-m": {
        "type": str,
        "default": "clip_vit_liaon",
        "choices": {
            "mobilenet_v3_small": "small MobileNetV3 model for CPU / low-memory GPU...",
            "mobilenet_v3_large": "large MobileNetV3 model for better accuracy, still fast...",
            "convnext_small": "ConvNeXt small model, good balance of speed and accuracy.",
            "regnet_y_400mf": "RegNetY 400MF, very fast, low accuracy.",
            "regnet_y_800mf": "RegNetY 800MF, fast, moderate accuracy.",
            "regnet_y_1_6gf": "RegNetY 1.6GF, balanced speed and accuracy.",
            "regnet_y_3_2gf": "RegNetY 3.2GF, good accuracy, moderate speed.",
            "regnet_y_8gf": "RegNetY 8GF, high accuracy, slower.",
            "regnet_y_16gf": "RegNetY 16GF, very high accuracy, needs more memory.",
            "regnet_y_32gf": "RegNetY 32GF, extremely high accuracy, high memory usage.",
            "regnet_y_128gf": "RegNetY 128GF, state-of-the-art, very high memory usage.",
            "clip_vit_large": "OpenAI CLIP ViT-Large (ViT-L/14), 336px, good accuracy, needs more memory.",
            "clip_vit_liaon": "LAION CLIP ViT-Huge (ViT-H/14), excellent accuracy.",
            "efficientnet_v2_s": "EfficientNetV2 Small, good balance.",
            "efficientnet_v2_m": "EfficientNetV2 Medium, better accuracy.",
            "efficientnet_v2_l": "EfficientNetV2 Large, high accuracy.",
            "clip_vit_liaon_mega": "LAION CLIP ViT-bigG (ViT-bigG/14), best accuracy, highest memory usage.",
        },
        "alt": ["--model", "--model_name"],
        "help": "Какая модель будет использоваться для извлечения фичей",
    },
    "--more_scan": {
        "action": "store_true",
        "help": "Использовать более качественный, но медленный режим извлечения признаков (анализ 6 вариантов изображения)."
    },
    "-i": {
        "type": str,
        "default": 'input_images',
        "alt": ["--input", "--input_folder"],
        "help": "Папка с исходными изображениями."
    },
    "-o": {
        "type": str,
        "default": 'sorted_images',
        "alt": ["--output", "--output_folder"],
        "help": "Папка для сохранения отсортированных изображений."
    },
    "--use_cpu": {
        "action": 'store_true',
        "alt": ["--cpu"],
        "help": "Использовать CPU вместо GPU."
    },
    # ------------------ Новые аргументы ------------------
    "--list_only": {
        "action": 'store_true',
        "help": "Не копировать файлы. Вместо этого сохранить TSV со списком в формате: path\\tindex\\tdistance (distance as index:dist,index:dist ...)."
    },
    "--tsv_neighbors": {
        "type": int,
        "default": 5,
        "help": "Сколько глобальных соседей выводить для каждой картинки (по умолчанию 5)."
    },
    "--batch_size": {
        "type": int,
        "default": 2048,
        "help": "Размер батча для операций поиска соседей в FAISS."
    },
    "--image_batch_size": {
        "type": int,
        "default": 1024,
        "help": "Количество файлов для обработки перед сохранением в БД."
    },
    "--feature_workers": {
        "type": int,
        "default": None,
        "help": ""
    },
    "--neighbors_k_limit": {
        "type": int,
        "default": 1024,
        "help": "Максимальное количество соседей для построения графа при сортировке."
    },
    "--lookahead": {
        "type": int,
        "default": 100,
        "help": "Глубина просмотра для оптимизированного DFS (0 для отключения)."
    },
    "--out_tsv": {
        "type": str,
        "default": "neighbor_list.tsv",
        "help": "Путь к выходному TSV-файлу (по умолчанию ./neighbor_list.tsv)."
    },
    "--find": {
        "nargs": "?",
        "const": "__PIPE__",
        "help": "Найти N ближайших соседей: если указан путь — ищет один файл, если без значения — работает как pipeline, читая пути из stdin."
    },
    "--index_file": {
        "type": str,
        "default": "faiss.index",
        "help": "Путь к файлу FAISS индекса для сохранения/загрузки."
    },
    "--find_neighbors": {
        "type": int,
        "default": 5,
        "help": "Сколько соседей возвращать для --find (по умолчанию 5)."
    },
    "--two_opt_shift": {
        "type": int,
        "default": 90,
        "help": "Сдвиг между блоками при пост-оптимизации 2-opt."
    },
    "--two_opt_block_size": {
        "type": int,
        "default": 100,
        "help": "Размер блока для пост-оптимизации 2-opt."
    },
    "--sort_optimizer": {
        "type": str,
        "default": "2opt",
        "help": "Алгоритм пост-оптимизации пути ('2opt' или другие в будущем)."
    },
    "--loglevel": {
        "type": str,
        "default": 'default',
        "choices": ["default", "error", "quiet", "debug"],
        "help": "Уровень детализации логирования."
    },
    "--find_result_type": {
        "type":str,
        "default":'both',
        "choices": ["indexed", "path", "both"],
        "help":"Формат вывода результатов поиска: 'indexed', 'path' или 'both'."
    },
    "--sort_strategy": {
        "type": str,
        "default": "farthest_insertion",
        "choices": ["dfs", "farthest_insertion", "christofides"],
        "help": "Алгоритм построения начального пути: 'dfs' (по умолчанию), 'farthest_insertion', 'christofides'."
    },
    "--list_objects": {
        "action": "store_true",
        "help": "List every file path currently stored in the features database."
    },
    "--move_db": {
        "nargs": 2,
        "metavar": ("OLD_ROOT", "NEW_ROOT"),
        "type": str,
        "help": "Move images from OLD_ROOT to NEW_ROOT and update their paths in the database."
    },
    "--cluster": {
        "action": "store_true",
        "help": "Group images into clusters based on feature distance threshold."
    },
    "--save_discarded": {
        "type": str,
        "default": "true",
        "metavar": "BOOL",
        "help": "Whether to save images that did not join any cluster. Accepts true/false values."
    },
    "--save_mode": {
        "type": str,
        "default": "default",
        "choices": ["default", "json", "print"],
        "help": "How to store clustering results: 'default' folders, 'json' file only, or 'print' to console."
    },
    "--cluster_algorithm": {
        "type": str,
        "default": "distance",
        "choices": ["distance", "dbscan", "hdbscan", "cc_graph", "mutual_graph", "graph"],
        "help": "Clustering algorithm: 'distance' (greedy overlap), 'dbscan' (density), 'hdbscan' (hierarchical density), 'cc_graph' (ε-graph connected components), or 'mutual_graph' (mutual ε-neighbors). 'graph' is kept as alias for 'cc_graph'."
    },
    "--threshold": {
        "type": float,
        "default": 0.25,
        "help": "Maximum distance between images to consider them similar for clustering."
    },
    "--similarity_percent": {
        "type": float,
        "default": 60.0,
        "help": "Percentage of existing cluster images that must be within threshold for a new image to join."
    },
    "--cluster_min_size": {
        "type": int,
        "default": 2,
        "help": "Minimum number of images required for a group to be treated as a cluster."
    },
    "--cluster_naming_mode": {
        "type": str,
        "default": "default",
        "choices": ["default", "distance", "distance_plus"],
        "help": "Controls how cluster folders and files are named: default numbering or distance-based variants."
    },
    "--cluster_pca": {
        "action": "store_true",
        "help": "Enable PCA pre-processing before clustering (dimensionality reduction + whitening)."
    },
    "--cluster_pca_components": {
        "type": int,
        "default": 256,
        "help": "Number of PCA components to keep (reduced dimension). Will be clipped to [1, min(n-1, D)]."
    },
    "--cluster_pca_whiten": {
        "type": str,
        "default": "true",
        "metavar": "BOOL",
        "help": "Whether to whiten PCA components (true/false). Whitening approximates Mahalanobis distance."
    },


}

GROUP_VALUE_SPLIT_PATTERN = re.compile(r":(?=[A-Za-z0-9_\-]+(?:=|$))")
TRUE_STRINGS = {"1", "true", "yes", "on"}
FALSE_STRINGS = {"0", "false", "no", "off"}

GROUPED_ARGUMENT_SPECS = {
    "cluster": {
        "options": ("--cluster",),
        "primary": "--cluster",
        "emit": "always",
        "allow_no_value": True,
        "subparams": {
            "threshold": {"arg": "--threshold"},
            "similarity_percent": {"arg": "--similarity_percent"},
            "percent": {"arg": "--similarity_percent"},
            "cluster_min_size": {"arg": "--cluster_min_size"},
            "min_size": {"arg": "--cluster_min_size"},
            "naming_mode": {"arg": "--cluster_naming_mode"},
            "cluster_naming_mode": {"arg": "--cluster_naming_mode"},
            "mode": {"arg": "--cluster_naming_mode"},
            "save_discarded": {"arg": "--save_discarded"},
            "save_mode": {"arg": "--save_mode"},
            "algorithm": {"arg": "--cluster_algorithm"},
            "cluster_algorithm": {"arg": "--cluster_algorithm"},
            "pca": {"arg": "--cluster_pca", "type": "bool"},
            "pca_components": {"arg": "--cluster_pca_components"},
            "pca_whiten": {"arg": "--cluster_pca_whiten"},
        },
    },
    "model": {
        "options": ("-m", "--model", "--model_name"),
        "primary": "--model",
        "emit": "when_self",
        "allow_no_value": True,
        "subparams": {
            "name": {"target": "self"},
            "model": {"target": "self"},
            "cpu": {"arg": "--use_cpu", "type": "bool"},
            "use_cpu": {"arg": "--use_cpu", "type": "bool"},
            "more_scan": {"arg": "--more_scan", "type": "bool"},
            "scan": {"arg": "--more_scan", "type": "bool"},
            "image_batch": {"arg": "--image_batch_size"},
            "image_batch_size": {"arg": "--image_batch_size"},
            "batch_size": {"arg": "--image_batch_size"},
            "feature_workers": {"arg": "--feature_workers"},
            "workers": {"arg": "--feature_workers"},
        },
    },
    "files": {
        "options": ("--files",),
        "primary": None,
        "emit": "never",
        "allow_no_value": False,
        "subparams": {
            "input": {"arg": "--input"},
            "input_folder": {"arg": "--input"},
            "output": {"arg": "--output"},
            "output_folder": {"arg": "--output"},
            "index": {"arg": "--index_file"},
            "index_file": {"arg": "--index_file"},
            "tsv": {"arg": "--out_tsv"},
            "out_tsv": {"arg": "--out_tsv"},
        },
    },
    "find": {
        "options": ("--find",),
        "primary": "--find",
        "emit": "always",
        "allow_no_value": True,
        "subparams": {
            "query": {"target": "self"},
            "path": {"target": "self"},
            "input": {"target": "self"},
            "neighbors": {"arg": "--find_neighbors"},
            "k": {"arg": "--find_neighbors"},
            "tsv_neighbors": {"arg": "--tsv_neighbors"},
            "batch_size": {"arg": "--batch_size"},
            "result_type": {"arg": "--find_result_type"},
            "format": {"arg": "--find_result_type"},
        },
    },
    "sorting": {
        "options": ("--sorting",),
        "primary": None,
        "emit": "never",
        "allow_no_value": False,
        "subparams": {
            "optimizer": {"arg": "--sort_optimizer"},
            "sort_optimizer": {"arg": "--sort_optimizer"},
            "strategy": {"arg": "--sort_strategy"},
            "sort_strategy": {"arg": "--sort_strategy"},
            "lookahead": {"arg": "--lookahead"},
            "neighbors_k_limit": {"arg": "--neighbors_k_limit"},
            "two_opt_shift": {"arg": "--two_opt_shift"},
            "two_opt_block_size": {"arg": "--two_opt_block_size"},
        },
    },
    "misc": {
        "options": ("--misc",),
        "primary": None,
        "emit": "never",
        "allow_no_value": False,
        "subparams": {
            "loglevel": {"arg": "--loglevel"},
            "list_only": {"arg": "--list_only", "type": "bool"},
            "list_objects": {"arg": "--list_objects", "type": "bool"},
        },
    },
    "database": {
        "options": ("--database",),
        "primary": None,
        "emit": "never",
        "allow_no_value": False,
        "subparams": {
            "move": {"arg": "--move_db", "type": "pair"},
            "move_db": {"arg": "--move_db", "type": "pair"},
            "list_objects": {"arg": "--list_objects", "type": "bool"},
        },
    },
}

OPTION_TO_GROUP = {
    option: group_key
    for group_key, spec in GROUPED_ARGUMENT_SPECS.items()
    for option in spec["options"]
}

GROUPED_USAGE_HELP = textwrap.dedent(
    """\
Grouped argument shortcuts:
  --cluster threshold=0.3:cluster_min_size=3
  --model name=clip_vit_liaon:cpu=true:feature_workers=8
  --files input=raw_data:output=sorted:tsv=neighbors.tsv
  --find query=example.jpg:neighbors=10:result_type=path
  --sorting strategy=dfs:optimizer=2opt
  --misc loglevel=debug:list_only=true
  --database move=old_root|new_root

Repeat a group flag to add more values or avoid ':' when the value already contains it (e.g. on Windows paths use multiple --files entries).
"""
)


def _fail_group_argument(message: str) -> None:
    """Emit a consistent error for grouped argument parsing."""
    sys.stderr.write(f"{message}\n")
    sys.exit(2)


def _split_group_items(raw: str) -> List[str]:
    """Split a raw group value into individual key=value fragments."""
    return [item.strip() for item in GROUP_VALUE_SPLIT_PATTERN.split(raw) if item.strip()]


def _parse_bool(value: str) -> bool:
    """Interpret a string as a boolean flag."""
    if value == "" or value is None:
        return True
    lowered = value.lower()
    if lowered in TRUE_STRINGS:
        return True
    if lowered in FALSE_STRINGS:
        return False
    _fail_group_argument(f"Unsupported boolean value '{value}'. Use one of {sorted(TRUE_STRINGS | FALSE_STRINGS)}.")
    return False


def _parse_pair(value: str, option_token: str, key: str) -> Tuple[str, str]:
    """Parse a pair value separated by '|', '->', or ','."""
    if not value:
        _fail_group_argument(f"Sub-argument '{key}' for {option_token} requires two values.")
    for separator in ("|", "->", ","):
        if separator in value:
            left, right = value.split(separator, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right
    _fail_group_argument(
        f"Sub-argument '{key}' for {option_token} expects two values separated by '|', '->', or ','."
    )
    return "", ""


def _expand_group_argument(option_token: str, raw_value: str, spec: dict) -> Tuple[List[str], List[str]]:
    """Expand grouped key=value entries into individual CLI tokens."""
    primary_values: List[str] = []
    extra_tokens: List[str] = []
    for item in _split_group_items(raw_value):
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            key, value = item, ""
        key = key.strip().lower()
        value = value.strip()
        if not key:
            _fail_group_argument(f"Invalid grouped argument fragment '{item}' for {option_token}.")
        rule = spec["subparams"].get(key)
        if rule is None:
            _fail_group_argument(f"Unknown sub-argument '{key}' for {option_token}.")
        target = rule.get("target")
        if target == "self":
            if not value:
                _fail_group_argument(f"Sub-argument '{key}' for {option_token} requires a value.")
            if primary_values:
                _fail_group_argument(f"Multiple values specified for {option_token}; only one is allowed.")
            primary_values.append(value)
            continue
        arg_name = rule["arg"]
        type_hint = rule.get("type")
        if type_hint == "bool":
            should_enable = _parse_bool(value)
            if should_enable:
                if arg_name not in extra_tokens:
                    extra_tokens.append(arg_name)
            else:
                extra_tokens = [token for token in extra_tokens if token != arg_name]
        elif type_hint == "pair":
            left, right = _parse_pair(value, option_token, key)
            extra_tokens.append(arg_name)
            extra_tokens.extend([left, right])
        else:
            if value == "":
                _fail_group_argument(f"Sub-argument '{key}' for {option_token} requires a value.")
            extra_tokens.append(arg_name)
            extra_tokens.append(value)
    return primary_values, extra_tokens


def _normalize_group_args(raw_args: List[str]) -> List[str]:
    """Translate grouped arguments into regular argparse-compatible tokens."""
    normalized: List[str] = []
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        group_key = OPTION_TO_GROUP.get(token)
        if group_key is None:
            normalized.append(token)
            index += 1
            continue

        spec = GROUPED_ARGUMENT_SPECS[group_key]
        next_token = raw_args[index + 1] if (index + 1) < len(raw_args) else None
        treat_as_group = next_token is not None and "=" in next_token
        if not treat_as_group:
            if spec.get("allow_no_value", False):
                primary = spec.get("primary") or token
                normalized.append(primary)
                index += 1
                continue
            _fail_group_argument(f"{token} requires at least one key=value pair.")

        primary_values, extra_tokens = _expand_group_argument(token, next_token, spec)
        emit_mode = spec.get("emit", "always")
        primary = spec.get("primary") or token
        if emit_mode == "always":
            normalized.append(primary)
        elif emit_mode == "when_self" and primary_values:
            normalized.append(primary)
        # emit == "never" -> skip adding the group flag itself

        normalized.extend(primary_values)
        normalized.extend(extra_tokens)
        index += 2
    return normalized

# --- Инициализация парсера и динамическое добавление аргументов ---
parser = argparse.ArgumentParser(
    description="Сортировка картинок по визуальной близости с выбором модели",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=GROUPED_USAGE_HELP,
)
for arg_name, config in PROGRAMM_ARGS.items():
    # Собираем все имена аргумента (основное + альтернативные)
    arg_names = [arg_name]
    if "alt" in config:
        arg_names.extend(config["alt"])
    
    # Удаляем ключ 'alt' из конфига, так как он не нужен для add_argument
    config_for_parser = {k: v for k, v in config.items() if k != "alt"}
    
    # Добавляем аргумент в парсер
    parser.add_argument(*arg_names, **config_for_parser)

# --- Парсинг аргументов ---
normalized_cli_args = _normalize_group_args(sys.argv[1:])
args = parser.parse_args(normalized_cli_args)

LOGGER = CustomLogger(level = LogLevel(args.loglevel))
CONFIG = Config(args)


# --- Конфигурации нейросетей ---
if CONFIG.model.use_cpu:
    DEVICE = torch.device("cpu")
    LOGGER.info("задано использование только процессора")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
