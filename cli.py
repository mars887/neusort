# ---------------------------------------------------------------------------- #
#                              Аргументы и Конфигурация                        #
# ---------------------------------------------------------------------------- #
import argparse
import torch


parser = argparse.ArgumentParser(
    description="Сортировка картинок по визуальной близости с выбором модели"
)
parser.add_argument(
    "-m",
    type=str,
    default='regnet_y_16gf',
    choices=["mobilenet_v3_small",
             "mobilenet_v3_large",
             "convnext_small", 
             "regnet_y_400mf",
             "regnet_y_800mf",
             "regnet_y_1_6gf",
             "regnet_y_3_2gf",
             "regnet_y_8gf",
             "regnet_y_16gf",
             "regnet_y_32gf",
             "regnet_y_128gf",
             "clip_vit_large",
             "clip_vit_liaon",
             "efficientnet_v2_s",
             "efficientnet_v2_m",
             "efficientnet_v2_l",
             "clip_vit_liaon_mega"],
    help="Какая модель будет использоваться для извлечения фичей",
)
 
parser.add_argument(
    "--more_scan",
    action='store_true',
    help="Использовать более качественный, но медленный режим извлечения признаков (анализ 6 вариантов изображения)."
)
parser.add_argument(
    "-i",
    type=str,
    default='input_images',
    help="Папка с исходными изображениями."
)
parser.add_argument(
    "-o",
    type=str,
    default='sorted_images',
    help="Папка для сохранения отсортированных изображений."
)

parser.add_argument(
    "--use_cpu",
    action='store_true',
    help="Использовать CPU вместо GPU."
)

# ------------------ Новые аргументы ------------------
parser.add_argument(
    "--list_only",
    action='store_true',
    help="Не копировать файлы. Вместо этого сохранить TSV со списком в формате: path\\tindex\\tdistance (distance as index:dist,index:dist ...)."
)

parser.add_argument(
    "--tsv_neighbors",
    type=int,
    default=5,
    help="Сколько глобальных соседей выводить для каждой картинки (по умолчанию 5)."
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=8192,
    help=""
)

parser.add_argument(
    "--image_batch_size",
    type=int,
    default=1024,
    help=""
)

parser.add_argument(
    "--neighbors_k_limit",
    type=int,
    default=1024,
    help=""
)

parser.add_argument(
    "--lookahead",
    type=int,
    default=100,
    help=""
)

parser.add_argument(
    "--out_tsv",
    type=str,
    default="neighbor_list.tsv",
    help="Путь к выходному TSV-файлу (по умолчанию ./neighbor_list.tsv)."
)

parser.add_argument(
    "--find",
    nargs="?",
    const="__PIPE__",
    help="Найти N ближайших соседей: если указан путь — ищет один файл, если без значения — работает как pipeline, читая пути из stdin."
)

parser.add_argument(
    "--index_file",
    type=str,
    default="faiss.index",
    help="Путь к файлу FAISS индекса для сохранения/загрузки."
)

parser.add_argument(
    "--find_neighbors",
    type=int,
    default=5,
    help="Сколько соседей возвращать для --find (по умолчанию 5)."
)

parser.add_argument(
    "--two_opt_shift",
    type=int,
    default=90,
    help=""
)

parser.add_argument(
    "--two_opt_block_size",
    type=int,
    default=100,
    help=""
)

parser.add_argument(
    "--sort_optimizer",
    type=str,
    default="2opt",
    help=""
)

parser.add_argument(
    "--loglevel",
    type=str,
    default='default',
    choices=["default","error","quiet"],
    help="Log level",
)




# ------------------------------------------------------


args = parser.parse_args()
ARG_MODEL_NAME = args.m
ARG_MORE_SCAN = args.more_scan
ARG_SRC_FOLDER = args.i
ARG_DST_FOLDER = args.o
ARG_LOG_LEVEL = args.loglevel
ARG_FIND = args.find
ARG_USE_CPU = args.use_cpu
ARG_FIND_NEIGHBORS = args.find_neighbors
ARG_INDEX_FILE = args.index_file
ARG_IMAGE_BATCH_SIZE = args.image_batch_size
ARG_LOOKAHEAD = args.lookahead
ARG_BATCH_SIZE = args.batch_size
ARG_SORT_OPTIMIZER = args.sort_optimizer
ARG_TWO_OPT_BLOCK_SIZE = args.two_opt_block_size
ARG_TWO_OPT_SHIFT = args.two_opt_shift
ARG_LIST_ONLY = args.list_only
ARG_OUT_TSV = args.out_tsv
ARG_NEIGHBORS_K_LIMIT = args.neighbors_k_limit
ARG_TSV_NEIGHBORS = args.tsv_neighbors

# --- Конфигурации нейросетей ---
if ARG_USE_CPU:
    DEVICE = torch.device("cpu")
    if ARG_LOG_LEVEL == "default": print("задано использование только процессора")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

