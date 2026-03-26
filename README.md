# Neusort v5

Neusort v5 сортирует, кластеризует и ищет похожие изображения по векторным признакам. Репозиторий строит эмбеддинги для файлов из входной папки, кэширует их в SQLite, а затем использует эти признаки для:

- сортировки изображений в визуально связную последовательность;
- кластеризации похожих изображений;
- поиска ближайших соседей по изображению, тексту или смешанному запросу.

Документация ниже составлена по текущему коду репозитория, а не только по `--help`.

## Быстрый старт

```powershell
python main.py --sorting -i input_images -o sorted_images
python main.py --cluster algorithm=dbscan:threshold=0.3 -i input_images -o clustered_images
python main.py --find path\\to\\query.jpg find_neighbors=10
python main.py --find "night city street" query_mode=text -m clip_vit_liaon
python main.py --index_only -i input_images
```

## Как устроен пайплайн

### 1. Извлечение и кэширование признаков

- Точка входа: `main.py`.
- Для тяжёлых задач (`--sorting`, `--cluster`, `--find`) проверяется существование входной папки.
- Имя базы признаков формируется как `features_db_<model>.sqlite` или `features_db_<model>_more_scan.sqlite`.
- В `database.py` проект:
  - создаёт таблицу `features(filename, features)` в SQLite;
  - сравнивает файлы на диске с уже сохранёнными путями;
  - считает признаки только для новых файлов;
  - сохраняет эмбеддинги как `float32` BLOB.

Важно:

- сканируется только верхний уровень входной папки;
- поддерживаемые расширения: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.jfif`;
- удалённые файлы не удаляются из БД физически, но игнорируются при загрузке признаков;
- при работе на GPU извлечение признаков принудительно переводится в последовательный режим, чтобы избежать конфликтов CUDA-потоков.

### 2. Извлечение признаков из изображения

`features.py` вычисляет один L2-нормализованный вектор на изображение.

- Для обычного режима берётся один центральный crop.
- Для `--more_scan` берётся несколько crop-областей, затем их признаки усредняются.
- Для `clip_*` используются отдельные кодовые пути с CLIP/OpenCLIP/Transformers-процессорами.
- Для остальных моделей используются torchvision или timm-модели с hook перед классификатором.

Важно: по факту `--more_scan` не включает рекурсивный обход папок. Этот флаг переключает многокроповый режим извлечения признаков и создаёт отдельный SQLite-кэш.

### 3. Сортировка

`core.run_sorting_pipeline()`:

1. гарантирует наличие актуального кэша признаков;
2. загружает признаки из SQLite;
3. строит FAISS `IndexFlatL2`;
4. запускает `sorting.sort_images()`.

Идея сортировки:

- строится приближённый граф соседства;
- основной порядок получается через эвристики ANN + MST;
- затем порядок локально улучшается;
- результат сохраняется в:
  - `sorted_images/` с префиксами `000_name.ext`;
  - `faiss.index`;
  - `faiss.index.order.npy`;
  - `faiss.index.paths.txt`.

Если включён `--list_only`, вместо копирования файлов формируется TSV-файл соседей (`--out_tsv`).

### 4. Кластеризация

`core.run_clustering_pipeline()`:

1. обновляет кэш признаков;
2. загружает признаки;
3. при `pca=true` применяет PCA + whitening;
4. выбирает алгоритм кластеризации;
5. при `enable_refine=true` выполняет пост-обработку кластеров;
6. экспортирует результат в выходную папку и/или `clusters.json`.

Поддерживаемые алгоритмы:

- `distance`
- `hdbscan`
- `dbscan`
- `cc_graph`
- `mutual_graph`
- `agglomerative`
- `agglomerative_complete`
- `optics`
- `snn`
- `rank_mutual`
- `adaptive_graph`

Общее правило:

- `threshold` в основном означает радиус/порог расстояния;
- `cluster_min_size` отбрасывает слишком маленькие кластеры;
- часть алгоритмов интерпретирует `threshold` по-своему, а `hdbscan` игнорирует его полностью.

### 5. Поиск

`core.run_search_pipeline()`:

1. загружает признаки из SQLite;
2. загружает уже сохранённый FAISS-индекс;
3. при необходимости загружает модель для внешнего query;
4. выполняет поиск ближайших соседей.

Режимы поиска:

- `image`: запрос по изображению;
- `text`: запрос по тексту;
- `image+text`: смешанный запрос.

Особенности:

- если запросное изображение уже есть в базе, переиспользуется готовый эмбеддинг;
- `text` и `image+text` доступны только для моделей `clip_*`;
- если `query_mode` не задан явно, код пытается определить его автоматически:
  - путь к существующему файлу -> `image` или `image+text`;
  - строка, не являющаяся файлом -> `text`;
  - строка вида `text | path/to/image` -> `image+text`;
- `--find` считается финальной задачей и всегда выполняется последней.

## Принцип работы CLI

CLI описан схемой в `cli.py`.

- Верхнеуровневые параметры передаются как обычные аргументы argparse.
- Подпараметры задач передаются одной строкой сразу после флага задачи.
- Разделитель подпараметров: `:`.
- Формат: `ключ=значение`.

Примеры:

```powershell
python main.py --cluster algorithm=dbscan:threshold=0.3:cluster_min_size=3
python main.py --sorting lookahead=64:sort_strategy=farthest_insertion
python main.py --find path\\to\\query.jpg find_neighbors=20:find_result_type=both
```

План выполнения строится по порядку аргументов, но с особенностью:

- `--sorting` и `--cluster` выполняются в том порядке, в котором указаны;
- `--move_db` и `--list_objects` выполняются как отдельные задачи;
- первый `--find` переносится в конец плана;
- если не передано ни одной задачи, по умолчанию запускается `--sorting`.

## Параметры верхнего уровня

| Параметр | По умолчанию | Назначение |
|---|---:|---|
| `-m`, `--model`, `--model_name` | `clip_vit_liaon` | Модель для извлечения признаков |
| `--more_scan` | `false` | Многокроповый режим извлечения признаков |
| `--feature_workers` | `os.cpu_count()` | Количество worker-потоков/задач при CPU-извлечении признаков |
| `--image_batch_size` | `1024` | Размер батча при обработке входных файлов |
| `-i`, `--input`, `--input_folder` | `input_images` | Входная папка с изображениями |
| `-o`, `--output`, `--output_folder` | `sorted_images` | Выходная папка |
| `--index_file` | `faiss.index` | Путь к FAISS-индексу |
| `--out_tsv` | `neighbor_list.tsv` | TSV-файл для списков соседей |
| `--use_cpu`, `--cpu` | `false` | Отключить GPU и работать только на CPU |
| `--loglevel` | `default` | `default`, `error`, `quiet`, `debug` |
| `--list_objects` | `false` | Вывести все пути, сохранённые в SQLite |
| `--move_db OLD_ROOT NEW_ROOT` | - | Переместить файлы и переписать пути в БД |
| `--print_params all|entered` | - | Показать эффективные или только введённые параметры |
| `--list_only` | `false` | Не копировать файлы; для сортировки формировать TSV, для кластеризации печатать/сводить без фактического экспорта |
| `--index_only` | `false` | Только вычислить/обновить кэш признаков и выйти |

## Подпараметры `--sorting`

Использование:

```powershell
python main.py --sorting lookahead=100:neighbors_k_limit=1024:sort_strategy=farthest_insertion
```

| Подпараметр | По умолчанию | Назначение |
|---|---:|---|
| `lookahead` | `100` | Глубина lookahead при обходе основного компонента |
| `neighbors_k_limit` | `1024` | Сколько ближайших соседей хранить для внутренних этапов сортировки |
| `two_opt_shift` | `90` | Размер окна смещения для локального 2-opt |
| `two_opt_block_size` | `100` | Размер блока для локальной оптимизации |
| `sort_optimizer` | `2opt` | Локальный оптимизатор после построения пути |
| `sort_strategy` | `farthest_insertion` | Основная стратегия обхода (`dfs` или `farthest_insertion`) |

## Подпараметры `--cluster`

Использование:

```powershell
python main.py --cluster algorithm=distance:threshold=0.35:cluster_min_size=2
```

| Подпараметр | По умолчанию | Назначение |
|---|---:|---|
| `algorithm` | `distance` | Алгоритм кластеризации |
| `threshold` | `0.35` | Порог расстояния / радиус / distance threshold в зависимости от алгоритма |
| `similarity_percent`, `percent` | `50.0` | Доля ближайших соседей для графовых вариантов |
| `cluster_min_size`, `min_size` | `2` | Минимальный размер валидного кластера |
| `cluster_naming_mode`, `naming_mode`, `mode` | `default` | Имена папок кластеров: `default`, `distance`, `distance_plus` |
| `save_discarded` | `true` | Сохранять ли нераспределённые изображения |
| `save_mode` | `default` | Формат экспорта: `default`, `json`, `print`, `group_filling` |
| `pca`, `cluster_pca` | `false` | Включить PCA + whitening перед кластеризацией |
| `group_filling_size`, `group_size`, `x` | `10` | Целевой размер группы для `save_mode=group_filling` |
| `cluster_splitting_mode`, `split_mode` | `recluster` | Способ деления слишком больших групп: `recluster` или `fi` |
| `similar_fill`, `similar_filling` | `false` | При заполнении групп брать элементы ближе к центроиду |
| `enable_refine`, `refine_clusters` | `false` | Пост-обработка кластеров: split/prune/garbage filter |

Замечания:

- `hdbscan` использует только `cluster_min_size` и не использует `threshold`;
- для `dbscan` `threshold` работает как `eps`;
- для `agglomerative` и `agglomerative_complete` `threshold` используется как `distance_threshold`;
- для `optics` `threshold` интерпретируется как `max_eps`;
- в коде также поддерживается `save_mode=cluster_sort`, хотя этот режим не показан в `--help`.

## Подпараметры `--find`

Использование:

```powershell
python main.py --find path\\to\\query.jpg find_neighbors=10:find_result_type=both
python main.py --find "night city street" query_mode=text -m clip_vit_liaon
python main.py --find "rainy night | path\\to\\query.jpg" query_mode=image+text:fusion_mode=directional
```

| Подпараметр | По умолчанию | Назначение |
|---|---:|---|
| `batch_size` | `2048` | Размер батча при глобальном k-NN |
| `find_neighbors`, `neighbors`, `k` | `5` | Сколько соседей вернуть |
| `find_result_type`, `result_type`, `format` | `both` | Что выводить: `indexed`, `path`, `both` |
| `tsv_neighbors` | `5` | Сколько соседей писать в TSV при `--list_only` |
| `backend` | `auto` | Принудительный backend поиска: `auto`, `regnet`, `clip` |
| `query_mode`, `mode` | `image` | `image`, `text`, `image+text` |
| `fusion_mode`, `fusion` | `simple` | Слияние image/text: `simple` или `directional` |
| `image_weight`, `w_img` | `0.5` | Вес изображения в `simple` fusion |
| `text_weight`, `w_txt` | `0.5` | Вес текста в `simple` fusion |
| `directional_alpha`, `alpha` | `0.7` | Сила смещения в `directional` fusion |
| `base_prompt`, `base_text` | `a photo on a road` | Базовый текст для directional fusion |

Режим pipeline:

```powershell
Get-Content queries.txt | python main.py --find
Get-Content text_queries.txt | python main.py --find __PIPE__ query_mode=text -m clip_vit_liaon
Get-Content mixed_queries.txt | python main.py --find __PIPE__
```

Для `image+text` каждая строка stdin должна иметь формат:

```text
текстовый_запрос | путь_к_файлу
```

## Поддерживаемые модели

### Torchvision

- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `convnext_small`
- `regnet_y_400mf`
- `regnet_y_800mf`
- `regnet_y_1_6gf`
- `regnet_y_3_2gf`
- `regnet_y_8gf`
- `regnet_y_16gf`
- `regnet_y_32gf`
- `regnet_y_128gf`
- `efficientnet_v2_s`
- `efficientnet_v2_m`
- `efficientnet_v2_l`

### CLIP / OpenCLIP

- `clip_vit_large`
- `clip_vit_liaon`
- `clip_vit_liaon_mega`

### timm / HF

- `anime_eva02_large`
- `anime_convnextv2_huge`
- `anime_swinv2_base_w8`
- `anime_caformer_b36`
- `anime_mobilenetv4_conv_aa_large`

## Выходные артефакты

### Кэш и индексы

- `features_db_<model>.sqlite` или `features_db_<model>_more_scan.sqlite`
- `faiss.index`
- `faiss.index.order.npy`
- `faiss.index.paths.txt`

### Сортировка

- папка `--output` с копиями файлов в новом порядке;
- либо TSV-файл `--out_tsv`, если включён `--list_only`.

### Кластеризация

- папка `--output` с подпапками кластеров;
- `clusters.json`;
- папка `unclustered/`, если есть отброшенные изображения и `save_discarded=true`.

### Поиск

- результат печатается в stdout;
- используется уже существующий FAISS-индекс, поэтому перед первым поиском нужно хотя бы один раз выполнить сортировку или иначе построить индекс.

## Нюансы текущей реализации

- `--find` требует не только SQLite-кэш, но и файлы `faiss.index`, `faiss.index.order.npy`, `faiss.index.paths.txt`.
- Поиск по тексту работает только с `clip_*`.
- `--more_scan` влияет на способ извлечения признаков и имя базы, а не на обход директорий.
- CLI показывает `save_mode` без `cluster_sort`, но код его принимает и умеет экспортировать.
- Включение `pca=true` доступно из CLI, но параметры `pca_components`, `pca_whiten`, `cluster_pairwise_limit`, `cluster_distance_chunk` сейчас зашиты как внутренние значения и не выведены как обычные CLI-флаги.

## Полезные сценарии

### Только подготовить кэш признаков

```powershell
python main.py --index_only -i input_images -m clip_vit_liaon
```

### Отсортировать изображения

```powershell
python main.py --sorting -i input_images -o sorted_images
```

### Кластеризовать с DBSCAN

```powershell
python main.py --cluster algorithm=dbscan:threshold=0.3:cluster_min_size=3 -i input_images -o clustered
```

### Кластеризовать и вывести только JSON

```powershell
python main.py --cluster algorithm=distance:save_mode=json -i input_images -o clustered
```

### Найти похожие изображения

```powershell
python main.py --find path\\to\\query.jpg find_neighbors=20:find_result_type=both
```

### Текстовый поиск по CLIP

```powershell
python main.py --find "red car at sunset" query_mode=text -m clip_vit_liaon
```
