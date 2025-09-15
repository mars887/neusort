import os
import faiss
import numpy as np
import torch
from tqdm.auto import tqdm

from cli import ARG_BATCH_SIZE, ARG_LOG_LEVEL
from faiss_io import load_faiss_index
from cli import LOGGER
from models import load_model


def compute_global_knn(feats: np.ndarray, k: int, batch_size: int = ARG_BATCH_SIZE, use_gpu: bool = False):
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

    for i in tqdm(range(0, n, batch_size), desc="  - Global k-NN search (batched)"):
        end = min(n, i + batch_size)
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

def find_and_print_neighbors_simple(index_file, feats, paths, target_path, k=5, use_gpu=False):
    """
    Находит k ближайших соседей для target_path и печатает одну строку:
    "sourcePath neighbor1Path neighbor1Dist neighbor2Path neighbor2Dist ..."
    feats: np.ndarray (N,D) — в порядке соответствующем списку paths
    paths: list[str] length N
    target_path: полный путь (или basename, если уникален)
    """
    # Проверки
    if not os.path.exists(index_file):
        LOGGER.error(f"! FAISS index file not found: {index_file}. Постройте индекс без --find.")
        return 2

    # Загрузим индекс (функция load_faiss_index предполагается в вашем скрипте)
    try:
        index = load_faiss_index(index_file, use_gpu=use_gpu)
    except Exception as e:
        LOGGER.error(f"! Error loading FAISS index: {e}")
        return 3

    # Попытаемся загрузить mapping faiss_pos -> original_idx
    order_map = None
    order_npy = index_file + ".order.npy"
    paths_txt = index_file + ".paths.txt"
    if os.path.exists(order_npy):
        try:
            order_map = np.load(order_npy)   # faiss_pos -> original_idx
        except Exception:
            order_map = None

    # Если нет order.npy попытаемся загрузить paths.txt (каждая строка — путь в порядке индекса)
    faiss_pos_to_path = None
    if order_map is not None:
        # order_map[p] = original_idx -> map to path
        faiss_pos_to_path = [ paths[int(orig_idx)] for orig_idx in order_map.tolist() ]
    elif os.path.exists(paths_txt):
        try:
            with open(paths_txt, "r", encoding="utf-8") as f:
                faiss_pos_to_path = [line.strip() for line in f]
        except Exception:
            faiss_pos_to_path = None
    else:
        # fallback: считаем, что порядок в индексе совпадает с порядком "paths" (как прочитан из БД)
        faiss_pos_to_path = paths[:]  # shallow copy

    # Проверка согласованности
    ntotal = index.ntotal
    if faiss_pos_to_path is None or len(faiss_pos_to_path) < ntotal:
        LOGGER.error("! Нельзя сопоставить позиции FAISS с путями к файлам (order.npy / paths.txt отсутствуют или корявы).")
        LOGGER.error("  Пожалуйста, перестройте индекс и сохраните mapping (see --list_only behaviour).")
        return 4

    # Найдём оригинальный индекс (в списке paths) для target_path
    target_orig_idx = None
    # exact match сначала
    for i,p in enumerate(paths):
        if p == target_path:
            target_orig_idx = i
            break
    if target_orig_idx is None:
        # попробуем по basename, если это уникально
        base = os.path.basename(target_path)
        matches = [i for i,p in enumerate(paths) if os.path.basename(p) == base]
        if len(matches) == 1:
            target_orig_idx = matches[0]
        elif len(matches) > 1:
            LOGGER.error(f"! Найдено несколько файлов с basename {base}. Укажите полный путь.")
            # выведем первые 20 совпадений для помощи
            for i in matches[:20]:
                LOGGER.error(f"  - {i}\t{paths[i]}")
            return 5
        else:
            LOGGER.error(f"! Файл '{target_path}' не найден в кэше/БД.")
            return 6

    # Найдём position в FAISS (если order_map сохранён)
    pos_in_faiss = None
    if order_map is not None:
        # order_map[p] = original_idx -> нужно найти p где original_idx == target_orig_idx
        arr = np.where(order_map == target_orig_idx)[0]
        if arr.size == 1:
            pos_in_faiss = int(arr[0])
        else:
            # не ожидаемо — скорее rebuild
            pos_in_faiss = None

    # Если pos_in_faiss остался неизвестным, попробуем найти по пути в faiss_pos_to_path
    if pos_in_faiss is None:
        try:
            pos_in_faiss = faiss_pos_to_path.index(paths[target_orig_idx])
        except ValueError:
            pos_in_faiss = None

    if pos_in_faiss is None:
        LOGGER.error("! Не удалось найти позицию файла в сохранённом mapping. Перестройте индекс (без --find).")
        return 7

    # готовим запрос
    n = feats.shape[0]
    kk = min(k + 1, n)  # +1 чтобы можно исключить саму точку
    q = feats[target_orig_idx:target_orig_idx+1].astype('float32', copy=False)

    # выполняем поиск
    D, I = index.search(q, kk)  # D — обычно квадраты L2 для IndexFlatL2
    # преобразуем расстояния (sqrt) — это корректно для L2-индексов
    with np.errstate(invalid='ignore'):
        D = np.sqrt(D)

    # формируем вывод: sourcePath neighbor1Path neighbor1Dist ...
    out_items = [paths[target_orig_idx]]
    for idx, dist in zip(I[0], D[0]):
        idx = int(idx)
        if idx == pos_in_faiss:
            continue
        # безопасно берём путь по позиции FAISS
        neigh_path = faiss_pos_to_path[idx]
        out_items.append(neigh_path)
        out_items.append(f"{float(dist):.6f}")
        if len(out_items) >= 1 + 2*k:
            break

    # объединяем и печатаем одной строкой
    print("Paths: " + " ".join(out_items))
    return 0

def print_neighbors_indexed(paths,orig_idx,pos,neighbors):
    print(f"Indexed: {paths[orig_idx]}\t{pos}\t{','.join(neighbors)}")

def find_neighbors_for_external_image(index, index_file, query_path, extract_feature_fn, model_ctor_fn, model_name,
                                      more_scan_flag, order_map, faiss_pos_to_path, k=5, use_gpu=False):
    """
    Ищет k соседей для картинки, которой нет в БД.
    - index: загруженный faiss индекс
    - index_file: путь к индексу (только для сообщений об ошибке)
    - query_path: путь к внешнему изображению
    - extract_feature_fn: функция extract_feature(path, model, hook, more_scan)
    - model_ctor_fn: функция load_model(name) -> (model, hook_blob) (мы передадим load_model)
    - model_name: имя модели (для load_model)
    - more_scan_flag: whether to compute more_scan
    - order_map: array faiss_pos -> original_idx (опционально)
    - faiss_pos_to_path: list pos -> path (обязательно желательно)
    - k: количество соседей
    """
    if not os.path.exists(query_path):
        LOGGER.error(f"! Внешний файл не найден: {query_path}")
        return 10

    # Загружаем модель один раз
    try:
        model, hook = load_model(model_name)
    except Exception as e:
        LOGGER.error(f"! Не удалось загрузить модель для извлечения признаков: {e}")
        return 11

    try:
        qfeat = extract_feature_fn(query_path, model, hook, more_scan=more_scan_flag)
        if qfeat is None:
            LOGGER.error(f"! Не удалось извлечь признаки для {query_path}")
            return 12
    except Exception as e:
        LOGGER.error(f"! Ошибка при извлечении признаков для {query_path}: {e}")
        return 13
    finally:
        # можем удалить модель из памяти
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    q = qfeat.astype('float32').reshape(1, -1)

    # Выполняем поиск
    try:
        kk = min(k, max(1, index.ntotal))
        D, I = index.search(q, kk)
        with np.errstate(invalid='ignore'):
            D = np.sqrt(D)
    except Exception as e:
        LOGGER.error(f"! Ошибка при поиске в FAISS: {e}")
        return 14

    # Формируем список соседей pos:dist и пути
    neighbor_entries = []
    neighbor_paths = []
    for idx, dist in zip(I[0], D[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(faiss_pos_to_path):
            continue
        neighbor_entries.append(f"{idx}:{float(dist):.6f}")
        neighbor_paths.append(faiss_pos_to_path[idx])
        if len(neighbor_entries) >= k:
            break

    # Вывод: external image не имеет позиции в индексе -> ставим -1
    out_line = f"{query_path}\t-1\t{','.join(neighbor_entries)}"
    # Печатаем краткую строку (по вашему формату)
    print(out_line)
    for p, e in zip(neighbor_paths, neighbor_entries):
        print(f"  -> {p}\t{e.split(':')[1]}")
    return 0


def find_and_print_neighbors_simple_extended(
    index,                   # загруженный faiss.Index
    faiss_pos_to_path,       # list: faiss_pos -> path
    order_map,               # np.array: faiss_pos -> original_idx
    feats,                   # np.ndarray всех фич в original order
    paths,                   # list всех путей в original order
    query_path,              # путь к картинке (может быть внешней)
    k=5,
    use_gpu=False,
    extract_feature_fn=None, # функция extract_feature(path, model, hook, more_scan)
    load_model_fn=None,      # функция load_model(model_name) -> (model, hook)
    model_name=None,
    more_scan=False,
    qvec=None,               # OPTIONAL: заранее вычисленный вектор (1,D) float32
    model=None,              # OPTIONAL: заранее загруженная модель (если хотим переиспользовать)
    hook=None,               # OPTIONAL: hook для модели
):
    """
    Работает и для внутренних (в БД), и для внешних изображений.
    Если qvec передан — используется он (никаких загрузок модели/извлечений).
    Если qvec не передан и model/hook переданы — использует их для extract_feature, но НЕ удаляет модель.
    Если ничего не передано — попытается загрузить модель через load_model_fn и после работы удалит её.
    """
    # Валидация
    if index is None or faiss_pos_to_path is None or order_map is None or feats is None or paths is None:
        raise ValueError("index, faiss_pos_to_path, order_map, feats, paths required")

    # 1) определим, есть ли изображение в базе
    orig_idx = None
    if query_path in paths:
        orig_idx = paths.index(query_path)
    else:
        base = os.path.basename(query_path)
        matches = [i for i,p in enumerate(paths) if os.path.basename(p) == base]
        if len(matches) == 1:
            orig_idx = matches[0]
        else:
            orig_idx = None

    # 2) подготовим qvec (если не передан)
    model_loaded_here = False
    if qvec is None:
        if orig_idx is not None:
            qvec = feats[orig_idx:orig_idx+1].astype('float32', copy=False)
        else:
            # внешний: попытаемся использовать переданную модель, иначе загрузим свою
            if model is None or hook is None:
                if load_model_fn is None:
                    LOGGER.error(f"{query_path}\t-1\tERROR_NO_MODEL_FUNC")
                    return
                try:
                    model, hook = load_model_fn(model_name)
                    model_loaded_here = True
                except Exception as e:
                    LOGGER.error(f"{query_path}\t-1\tERROR_LOAD_MODEL:{e}")
                    return
            # извлекаем признаки
            try:
                qfeat = extract_feature_fn(query_path, model, hook, more_scan=more_scan)
                if qfeat is None:
                    LOGGER.error(f"{query_path}\t-1\tERROR_EXTRACT")
                    # если модель была загружена здесь — освободим
                    if model_loaded_here:
                        try:
                            del model
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                        except: pass
                    return
                qvec = qfeat.astype('float32').reshape(1, -1)
            except Exception as e:
                LOGGER.error(f"{query_path}\t-1\tERROR_EXTRACT:{e}")
                if model_loaded_here:
                    try:
                        del model
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    except: pass
                return
    # если модель была загружена внутри — не удаляем её до окончания работы функции (ниже)
    # 3) Поиск в FAISS
    n = max(1, index.ntotal)
    kk = min(k + (1 if orig_idx is not None else 0), n)
    try:
        D, I = index.search(qvec, kk)
    except Exception as e:
        LOGGER.error(f"{query_path}\t-1\tERROR_FAISS:{e}")
        if model_loaded_here:
            try:
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except: pass
        return

    with np.errstate(invalid='ignore'):
        try: D = np.sqrt(D)
        except: pass

    # 4) формируем вывод: source neighborPath neighborDist ...
    out_parts = [query_path]
    # при наличии orig_idx — исключаем саму точку
    pos_in_faiss_src = None
    if orig_idx is not None:
        pos_arr = np.where(order_map == orig_idx)[0]
        if pos_arr.size > 0:
            pos_in_faiss_src = int(pos_arr[0])

    for idx_val, dist_val in zip(I[0], D[0]):
        if int(idx_val) < 0: 
            continue
        faiss_pos = int(idx_val)
        if pos_in_faiss_src is not None and faiss_pos == pos_in_faiss_src:
            continue
        if faiss_pos < 0 or faiss_pos >= len(faiss_pos_to_path):
            continue
        neighbor_path = faiss_pos_to_path[faiss_pos]
        out_parts.append(neighbor_path)
        out_parts.append(f"{float(dist_val):.6f}")
        if len(out_parts) >= 1 + 2 * k:
            break

    # если модель загружалась здесь — освобождаем
    if model_loaded_here:
        try:
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception:
            pass

    print(" ".join(out_parts))
    return