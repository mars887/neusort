import os
import sys
import faiss
import numpy as np
import torch
from tqdm.auto import tqdm

from database import load_features_from_db
from faiss_io import load_faiss_index
from cli import LOGGER
from features import extract_feature
from models import load_model
from config import Config     


def compute_global_knn(feats: np.ndarray, k: int, config: Config, use_gpu: bool = False):
    """
    Для каждого вектора из feats возвращает k ближайших соседей по всей базе (по L2).
    Возвращает два массива:
      - knn_idxs: shape (n, k) — индексы соседей (оригинальные индексы в feats), -1 если нет
      - knn_dists: shape (n, k) — расстояния (L2, не квадраты), np.inf если нет
    По умолчанию выполняется батчевый поиск, использующий FAISS IndexFlatL2.
    """
    n, d = feats.shape
    feats_f32 = feats.astype('float32', copy=False)

    # Создаем индекс FAISS
    index = faiss.IndexFlatL2(d)
    if use_gpu and torch is not None and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            LOGGER.info(f"  - Предупреждение: не удалось перенести FAISS на GPU: {e}. Будет использован CPU.")
    index.add(feats_f32)

    knn_k = min(k + 1, n)  # ищем k+1 чтобы исключить саму точку
    knn_idxs = np.full((n, k), -1, dtype=np.int64)
    knn_dists = np.full((n, k), np.inf, dtype=np.float32)

    for i in tqdm(range(0, n, config.search.global_knn_batch_size), desc="  - Global k-NN search (batched)"):
        end = min(n, i + config.search.global_knn_batch_size)
        D, I = index.search(feats_f32[i:end], knn_k)  # D: квадраты L2
        # Для каждой строки удаляем саму точку и берем первые k
        for r in range(end - i):
            global_idx = i + r
            neigh_idx_row = I[r]
            neigh_dist_row = D[r]
            # Удаляем саму точку (если есть)
            mask = neigh_idx_row != global_idx
            filtered_idxs = neigh_idx_row[mask]
            filtered_dists = neigh_dist_row[mask]
            take = min(k, filtered_idxs.shape[0])
            if take > 0:
                knn_idxs[global_idx, :take] = filtered_idxs[:take]
                # sqrt расстояний (FAISS вернул квадраты), сохраняем L2
                knn_dists[global_idx, :take] = np.sqrt(filtered_dists[:take])
            # если take < k — оставшиеся значения останутся -1 / inf
    return knn_idxs, knn_dists

def find_neighbors(
    index,                   # загруженный faiss.Index
    faiss_pos_to_path,       # list: faiss_pos -> path
    order_map,               # np.array: faiss_pos -> original_idx
    feats,                   # np.ndarray всех фич в original order
    paths,                   # list всех путей в original order
    query_path,              # путь к картинке (может быть внешней)
    k=5,
    extract_feature_fn=None, # функция extract_feature(path, model, hook, more_scan)
    load_model_fn=None,      # функция load_model(model_name) -> (model, hook)
    model_name=None,
    more_scan=False,
    qvec=None,               # OPTIONAL: заранее вычисленный вектор (1,D) float32
    model=None,              # OPTIONAL: заранее загруженная модель
    hook=None,               # OPTIONAL: hook для модели
    config=None              # Добавляем config для доступа к настройкам
):
    """
    Универсальная функция для поиска k ближайших соседей.
    Работает как для изображений в базе данных, так и для внешних файлов.
    Возвращает структурированный результат, а не печатает его.

    Returns:
        dict: {
            'query_path': str,          # Путь к запрашиваемому изображению
            'is_external': bool,        # True, если изображение не в базе данных
            'original_index': int,      # Индекс в исходном списке paths (-1 для внешних)
            'faiss_position': int,      # Позиция в FAISS индексе (-1 для внешних или если не найдена)
            'neighbors': list[dict]     # Список соседей [{'index': int, 'distance': float, 'path': str}, ...]
        }
        или None в случае ошибки.
    """
    # Валидация
    if index is None or faiss_pos_to_path is None or order_map is None or feats is None or paths is None:
        LOGGER.error("Ошибка: отсутствуют необходимые данные (index, faiss_pos_to_path, order_map, feats, paths).")
        return None

    # 1) Определяем, есть ли изображение в базе
    orig_idx = None
    if query_path in paths:
        orig_idx = paths.index(query_path)
    else:
        base = os.path.basename(query_path)
        matches = [i for i, p in enumerate(paths) if os.path.basename(p) == base]
        if len(matches) == 1:
            orig_idx = matches[0]
        else:
            orig_idx = None  # Внешнее изображение

    is_external = (orig_idx is None)

    # 2) Подготавливаем вектор запроса (qvec)
    model_loaded_here = False
    if qvec is None:
        if not is_external:
            # Изображение в базе - берем готовый вектор
            qvec = feats[orig_idx:orig_idx + 1].astype('float32', copy=False)
        else:
            # Внешнее изображение - извлекаем признаки
            if model is None or hook is None:
                if load_model_fn is None:
                    LOGGER.error(f"{query_path}\t-1\tERROR_NO_MODEL_FUNC")
                    return None
                try:
                    model, hook = load_model_fn(model_name)
                    model_loaded_here = True
                except Exception as e:
                    LOGGER.error(f"{query_path}\t-1\tERROR_LOAD_MODEL:{e}")
                    return None

            try:
                qfeat = extract_feature_fn(query_path, model, hook, config)
                if qfeat is None:
                    LOGGER.error(f"{query_path}\t-1\tERROR_EXTRACT")
                    if model_loaded_here:
                        try:
                            del model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except:
                            pass
                    return None
                qvec = qfeat.astype('float32').reshape(1, -1)
            except Exception as e:
                LOGGER.error(f"{query_path}\t-1\tERROR_EXTRACT:{e}")
                if model_loaded_here:
                    try:
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                return None

    # Если модель была загружена внутри — освобождаем ее после поиска
    try:
        # 3) Поиск в FAISS
        n = max(1, index.ntotal)
        # Для внешних изображений мы ищем k соседей, для внутренних - k+1 (чтобы исключить само изображение)
        kk = min(k + (1 if not is_external else 0), n)
        D, I = index.search(qvec, kk)
        D = np.sqrt(D)  # Преобразуем квадраты расстояний в L2

        # 4) Определяем позицию в FAISS для внутренних изображений
        faiss_pos_src = -1
        if not is_external and order_map is not None:
            pos_arr = np.where(order_map == orig_idx)[0]
            if pos_arr.size > 0:
                faiss_pos_src = int(pos_arr[0])

        # 5) Формируем список соседей
        neighbors = []
        for idx_val, dist_val in zip(I[0], D[0]):
            faiss_pos = int(idx_val)
            # Пропускаем само изображение, если оно в базе
            if not is_external and faiss_pos == faiss_pos_src:
                continue
            if faiss_pos < 0 or faiss_pos >= len(faiss_pos_to_path):
                continue

            neighbor_path = faiss_pos_to_path[faiss_pos]
            # Находим original index соседа, если он есть в базе
            neighbor_orig_idx = -1
            if not is_external:  # Только если запрос был внутренним, иначе маппинг может быть неполным
                neighbor_orig_arr = np.where(order_map == faiss_pos)[0]
                if neighbor_orig_arr.size > 0:
                    neighbor_orig_idx = int(neighbor_orig_arr[0])

            neighbors.append({
                'faiss_index': faiss_pos,
                'original_index': neighbor_orig_idx,
                'distance': float(dist_val),
                'path': neighbor_path
            })

            if len(neighbors) >= k:
                break

        # 6) Возвращаем структурированный результат
        result = {
            'query_path': query_path,
            'is_external': is_external,
            'original_index': orig_idx if not is_external else -1,
            'faiss_position': faiss_pos_src if not is_external else -1,
            'neighbors': neighbors
        }
        return result

    finally:
        if model_loaded_here:
            try:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

def format_neighbors_output(result, output_format='both'):
    """
    Форматирует результат поиска соседей в читаемые строки.

    Args:
        result (dict): Результат функции `find_neighbors`.
        output_format (str): 'indexed', 'path', или 'both'.

    Returns:
        list[str]: Список строк для вывода.
    """
    if result is None:
        return []

    output_lines = []

    query_path = result['query_path']
    is_external = result['is_external']
    orig_idx = result['original_index']
    faiss_pos = result['faiss_position']
    neighbors = result['neighbors']

    # Формат "Indexed"
    if output_format in ['indexed', 'both']:
        if is_external:
            # Для внешнего: позиция -1, соседи в формате faiss_index:distance
            neighbor_str = ','.join([f"{n['faiss_index']}:{n['distance']:.6f}" for n in neighbors])
            line = f"Indexed: {query_path}\t-1\t{neighbor_str}"
        else:
            # Для внутреннего: используем позицию в FAISS
            neighbor_str = ','.join([f"{n['faiss_index']}:{n['distance']:.6f}" for n in neighbors])
            line = f"Indexed: {query_path}\t{faiss_pos}\t{neighbor_str}"
        output_lines.append(line)

    # Формат "Path"
    if output_format in ['path', 'both']:
        parts = [query_path]
        for n in neighbors:
            parts.append(n['path'])
            parts.append(f"{n['distance']:.6f}")
        line = "Paths: " + " ".join(parts)
        output_lines.append(line)

    return output_lines

def handle_search_pipeline(config: Config, feats, paths):
    use_gpu_faiss = (not config.model.use_cpu) and torch.cuda.is_available()

    # --- Подготовка: Загрузка индекса и вспомогательных данных ---
    if not os.path.exists(config.files.index_file):
        LOGGER.error(f"! Индекс не найден: {config.files.index_file}. Сначала запустите без --find, чтобы его построить.")
        exit(2)

    index = load_faiss_index(config)
    
    # Загрузка mapping файлов
    order_npy_path = config.files.index_file + ".order.npy"
    paths_txt_path = config.files.index_file + ".paths.txt"
    
    if not (os.path.exists(order_npy_path) and os.path.exists(paths_txt_path)):
        LOGGER.error("! Для поиска требуются файлы .order.npy и .paths.txt. Перестройте индекс.")
        exit(2)

    order_map = np.load(order_npy_path)  # faiss_pos -> original_idx
    with open(paths_txt_path, "r", encoding="utf-8") as f:
        faiss_pos_to_path = [line.strip() for line in f]

    # Загрузим модель и hook один раз для всех внешних изображений (если понадобится)
    model, hook = None, None
    if config.search.find == "__PIPE__" or config.search.find is not None:
        try:
            model, hook = load_model(config.model.model_name)
        except Exception as e:
            LOGGER.error(f"! Не удалось загрузить модель для поиска: {e}")
            exit(11)

    # --- Вспомогательная функция для поиска и вывода ---
    def process_single_query(query_path: str):
        """Обрабатывает один запрос поиска и выводит результат."""
        result = find_neighbors(
            index=index,
            faiss_pos_to_path=faiss_pos_to_path,
            order_map=order_map,
            feats=feats,
            paths=paths,
            query_path=query_path,
            k=config.search.find_neighbors,
            extract_feature_fn=extract_feature,
            load_model_fn=load_model,
            model_name=config.model.model_name,
            more_scan=config.model.more_scan,
            model=model,  # Переиспользуем модель, если она загружена
            hook=hook,
            config=config
        )

        if result is None:
            LOGGER.error(f"Не удалось найти соседей для: {query_path}")
            return

        # Форматируем и выводим результат в соответствии с настройкой
        output_lines = format_neighbors_output(result, output_format=config.search.find_result_type)
        for line in output_lines:
            print(line)

    # --- Режим pipeline: чтение путей из stdin ---
    if config.search.find == "__PIPE__":
        LOGGER.info("Pipeline mode started. Enter image paths (Ctrl+D/Ctrl+C to stop):")
        for line in sys.stdin:
            query_path = line.strip()
            if query_path:
                process_single_query(query_path)
    # --- Режим одиночного запроса ---
    else:
        target_path = config.search.find
        process_single_query(target_path)

    # --- Очистка ресурсов ---
    if model is not None:
        try:
            del model, hook
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            LOGGER.error(f"Ошибка при освобождении памяти модели: {e}")