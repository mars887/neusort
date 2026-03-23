import os
import sys
from typing import Optional

import faiss
import numpy as np
import torch
import runtime_state as runtime
from tqdm.auto import tqdm

from faiss_io import load_faiss_index
from clip_manager import CLIP_PROCESSOR_MANAGER
from features import extract_feature
from models import load_model
from config import Config


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32, copy=False)


def _normalize_query_path(path: str) -> str:
    """
    Normalize user-provided paths by:
    - stripping surrounding quotes
    - normalizing separators for the current OS
    """
    if not path:
        return path
    p = path.strip()
    if len(p) >= 2 and p[0] == p[-1] and p[0] in ("'", '"'):
        p = p[1:-1].strip()
    try:
        p = os.path.normpath(p)
    except Exception:
        pass
    return p


def _match_existing_index(query_path: str, paths) -> Optional[int]:
    if not query_path:
        return None
    normalized_query = _normalize_query_path(query_path)
    if normalized_query in paths:
        return paths.index(normalized_query)
    base = os.path.basename(normalized_query)
    matches = [i for i, p in enumerate(paths) if os.path.basename(p) == base]
    if len(matches) == 1:
        return matches[0]
    return None


def _encode_clip_text_feature(model, text: str) -> Optional[np.ndarray]:
    if not text:
        runtime.LOGGER.error("Текстовый запрос не задан.")
        return None
    text_processor = CLIP_PROCESSOR_MANAGER.text_processor
    if text_processor is None:
        runtime.LOGGER.error("CLIP текстовый процессор не инициализирован. Загрузите CLIP модель.")
        return None
    clip_inner = getattr(model, "inner", model)
    try:
        with torch.no_grad():
            if hasattr(clip_inner, "encode_text"):
                tokens = text_processor(text)
                if isinstance(tokens, dict):
                    tokens = tokens.get("input_ids") or next(iter(tokens.values()))
                tokens = tokens.to(runtime.DEVICE)
                feat = clip_inner.encode_text(tokens)
            elif hasattr(clip_inner, "get_text_features"):
                inputs = text_processor(
                    text=[text], return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(runtime.DEVICE) for k, v in inputs.items()}
                feat = clip_inner.get_text_features(**inputs)
            else:
                runtime.LOGGER.error("Загруженная модель не поддерживает текстовые запросы.")
                return None
    except Exception as exc:
        runtime.LOGGER.error(f"Не удалось получить CLIP текстовый эмбеддинг: {exc}")
        return None
    vec = feat.detach().cpu().numpy().reshape(-1)
    return _normalize_vec(vec)


def _get_image_embedding(query_path: str, feats, paths, model, hook, config: Config) -> Optional[np.ndarray]:
    clean_path = _normalize_query_path(query_path)
    idx = _match_existing_index(clean_path, paths)
    if idx is not None:
        return _normalize_vec(feats[idx].astype(np.float32, copy=False))
    try:
        feat = extract_feature(clean_path, model, hook, config)
    except Exception as exc:
        runtime.LOGGER.error(f"Не удалось извлечь признаки для {query_path}: {exc}")
        return None
    if feat is None:
        runtime.LOGGER.error(f"Не удалось извлечь признаки для {query_path}.")
        return None
    return _normalize_vec(feat.astype(np.float32, copy=False))


def _build_image_text_query(
    image_path: str,
    text: str,
    model,
    hook,
    feats,
    paths,
    config: Config,
    fusion_mode: str,
    img_weight: float,
    txt_weight: float,
    directional_alpha: float,
    base_text_vec: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    image_vec = _get_image_embedding(image_path, feats, paths, model, hook, config)
    if image_vec is None:
        return None
    text_vec = _encode_clip_text_feature(model, text)
    if text_vec is None:
        return None
    if fusion_mode == "directional":
        base_vec = base_text_vec
        if base_vec is None:
            runtime.LOGGER.warning("Базовое текстовое описание не задано, используем текст запроса как базу.")
            base_vec = text_vec
        direction = text_vec - base_vec
        norm = np.linalg.norm(direction)
        if norm == 0:
            direction = text_vec
        else:
            direction = direction / norm
        query_vec = _normalize_vec(image_vec + directional_alpha * direction)
    else:
        w_img = max(0.0, float(img_weight))
        w_txt = max(0.0, float(txt_weight))
        if w_img == 0 and w_txt == 0:
            w_img = w_txt = 0.5
        query_vec = _normalize_vec(w_img * image_vec + w_txt * text_vec)
    return query_vec


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
            runtime.LOGGER.info(f"  - Предупреждение: не удалось перенести FAISS на GPU: {e}. Будет использован CPU.")
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
        runtime.LOGGER.error("Ошибка: отсутствуют необходимые данные (index, faiss_pos_to_path, order_map, feats, paths).")
        return None

    # 1) Определяем, есть ли изображение в базе
    normalized_query_path = _normalize_query_path(query_path)
    orig_idx = _match_existing_index(normalized_query_path, paths)

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
                    runtime.LOGGER.error(f"{query_path}\t-1\tERROR_NO_MODEL_FUNC")
                    return None
                try:
                    model, hook = load_model_fn(model_name)
                    model_loaded_here = True
                except Exception as e:
                    runtime.LOGGER.error(f"{query_path}\t-1\tERROR_LOAD_MODEL:{e}")
                    return None

            try:
                qfeat = extract_feature_fn(normalized_query_path, model, hook, config)
                if qfeat is None:
                    runtime.LOGGER.error(f"{query_path}\t-1\tERROR_EXTRACT")
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
                runtime.LOGGER.error(f"{query_path}\t-1\tERROR_EXTRACT:{e}")
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
            neighbor_orig_idx = int(order_map[faiss_pos])

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
    query_mode = getattr(config.search, "query_mode", "image").lower()
    fusion_mode = getattr(config.search, "fusion_mode", "simple").lower()
    requires_text = query_mode in {"text", "image+text"}
    if requires_text and not config.model.model_name.startswith("clip_"):
        runtime.LOGGER.error("Текстовые запросы доступны только для CLIP-моделей. Укажите модель clip_* и пересчитайте признаки.")
        return

    # --- Подготовка: Загрузка индекса и вспомогательных данных ---
    if not os.path.exists(config.files.index_file):
        runtime.LOGGER.error(f"! Индекс не найден: {config.files.index_file}. Сначала запустите без --find, чтобы его построить.")
        return

    index = load_faiss_index(config)
    
    # Загрузка mapping файлов
    order_npy_path = config.files.index_file + ".order.npy"
    paths_txt_path = config.files.index_file + ".paths.txt"
    
    if not (os.path.exists(order_npy_path) and os.path.exists(paths_txt_path)):
        runtime.LOGGER.error("! Для поиска требуются файлы .order.npy и .paths.txt. Перестройте индекс.")
        return

    order_map = np.load(order_npy_path)  # faiss_pos -> original_idx
    with open(paths_txt_path, "r", encoding="utf-8") as f:
        faiss_pos_to_path = [line.strip() for line in f]

    model, hook = None, None
    def ensure_model_loaded() -> bool:
        nonlocal model, hook
        if model is not None and hook is not None:
            return True
        try:
            model, hook = load_model(config.model.model_name)
            return True
        except Exception as e:
            runtime.LOGGER.error(f"! Не удалось загрузить модель для поиска: {e}")
            return False

    if requires_text and not ensure_model_loaded():
        return

    base_text_vec = None
    if requires_text and fusion_mode == "directional":
        base_prompt = (getattr(config.search, "base_prompt", "") or "").strip()
        if base_prompt:
            base_text_vec = _encode_clip_text_feature(model, base_prompt)
            if base_text_vec is None:
                runtime.LOGGER.warning("Не удалось получить базовый текстовый эмбеддинг, будет использоваться текст запроса.")

    # --- Вспомогательная функция для поиска и вывода ---
    def process_single_query(query_input: str, text_override: Optional[str] = None):
        """Обрабатывает один запрос поиска и выводит результат."""
        qvec = None
        query_label = query_input

        if query_mode == "text":
            if not ensure_model_loaded():
                return
            actual_text = text_override
            if actual_text is None or actual_text == "":
                actual_text = config.search.query_text or query_input
            if not actual_text:
                runtime.LOGGER.error("Для текстового поиска нужно задать --query_text или строку в качестве значения --find.")
                return
            text_vec = _encode_clip_text_feature(model, actual_text)
            if text_vec is None:
                return
            qvec = text_vec.reshape(1, -1)
            query_label = f"[text] {actual_text}"
        elif query_mode == "image+text":
            if not ensure_model_loaded():
                return
            if not query_input:
                runtime.LOGGER.error("Для режима image+text необходимо указать путь к изображению.")
                return
            actual_text = text_override if text_override not in (None, "") else config.search.query_text
            if not actual_text:
                runtime.LOGGER.error("Для режима image+text добавьте описание через --query_text или передавайте его через pipeline.")
                return
            image_path = _normalize_query_path(query_input)
            query_vec = _build_image_text_query(
                image_path,
                actual_text,
                model,
                hook,
                feats,
                paths,
                config,
                fusion_mode,
                getattr(config.search, "image_weight", 0.5),
                getattr(config.search, "text_weight", 0.5),
                getattr(config.search, "directional_alpha", 0.7),
                base_text_vec,
            )
            if query_vec is None:
                return
            qvec = query_vec.reshape(1, -1)
            query_label = image_path
        else:
            # image-only mode: normalize possible quoted / mixed-separator path
            query_label = _normalize_query_path(query_input)
            if _match_existing_index(query_label, paths) is None and not ensure_model_loaded():
                return

        result = find_neighbors(
            index=index,
            faiss_pos_to_path=faiss_pos_to_path,
            order_map=order_map,
            feats=feats,
            paths=paths,
            query_path=query_label,
            k=config.search.find_neighbors,
            extract_feature_fn=extract_feature,
            load_model_fn=load_model,
            model_name=config.model.model_name,
            more_scan=config.model.more_scan,
            model=model,  # Переиспользуем модель, если она загружена
            hook=hook,
            config=config,
            qvec=qvec.astype(np.float32) if qvec is not None else None,
        )

        if result is None:
            runtime.LOGGER.error(f"Не удалось найти соседей для: {query_label}")
            return

        # Форматируем и выводим результат в соответствии с настройкой
        output_lines = format_neighbors_output(result, output_format=config.search.find_result_type)
        for line in output_lines:
            print(line)

    # --- Режим pipeline: чтение путей из stdin ---
    if config.search.find == "__PIPE__":
        if query_mode == "text":
            runtime.LOGGER.info("Pipeline text mode: вводите по одному текстовому запросу на строку (Ctrl+D/Ctrl+C для выхода).")
        elif query_mode == "image+text":
            runtime.LOGGER.info("Pipeline image+text: каждая строка должна быть 'путь\tописание'.")
        else:
            runtime.LOGGER.info("Pipeline mode started. Enter image paths (Ctrl+D/Ctrl+C to stop):")
        for raw_line in sys.stdin:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if query_mode == "text":
                process_single_query(line, text_override=line)
            elif query_mode == "image+text":
                if "\t" not in line:
                    runtime.LOGGER.error("Неверный формат строки. Используйте 'путь\tописание'.")
                    continue
                path_part, text_part = line.split("\t", 1)
                process_single_query(path_part.strip(), text_override=text_part.strip())
            else:
                process_single_query(line.strip())
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
            runtime.LOGGER.error(f"Ошибка при освобождении памяти модели: {e}")
