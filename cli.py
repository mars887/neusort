# ---------------------------------------------------------------------------- #
#                              Аргументы и Конфигурация                        #
# ---------------------------------------------------------------------------- #
import argparse
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
        "default": 8192,
        "help": "Размер батча для операций поиска соседей в FAISS."
    },
    "--image_batch_size": {
        "type": int,
        "default": 1024,
        "help": "Количество файлов для обработки перед сохранением в БД."
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
        "choices": ["default", "error", "quiet"],
        "help": "Уровень детализации логирования."
    }
}

# --- Инициализация парсера и динамическое добавление аргументов ---
parser = argparse.ArgumentParser(
    description="Сортировка картинок по визуальной близости с выбором модели"
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
args = parser.parse_args()

LOGGER = CustomLogger(level = LogLevel(args.loglevel))
CONFIG = Config(args)


# --- Конфигурации нейросетей ---
if CONFIG.model.use_cpu:
    DEVICE = torch.device("cpu")
    LOGGER.info("задано использование только процессора")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")