
import os
import sys

import faiss
import numpy as np
import torch

from database import load_features_from_db, process_and_cache_features
from faiss_io import load_faiss_index
from features import extract_feature
from models import load_model
from search import find_and_print_neighbors_simple, find_and_print_neighbors_simple_extended, print_neighbors_indexed
from sorting import sort_images

from cli import LOGGER
from cli import CONFIG

# ---------------------------------------------------------------------------- #
#                                    Main                                      #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # --- ЛОГИКА СОЗДАНИЯ ИМЕНИ БАЗЫ ДАННЫХ ---
    safe_model_name = CONFIG.model.model_name.replace("/", "_")
    scan_suffix = "_more_scan" if CONFIG.model.more_scan else ""
    DB_FILE = f"features_db_{safe_model_name}{scan_suffix}.sqlite"

    # Убедимся, что папка с исходниками есть
    if not os.path.isdir(CONFIG.files.src_folder):
        LOGGER.error(f"Создайте папку {CONFIG.files.src_folder} и положите туда картинки.")
        os.makedirs(CONFIG.files.src_folder, exist_ok=True)
        exit(1)

    # 2. Создаем или обновляем базу данных с признаками
    # Эта функция сама найдет новые файлы и обработает только их
    process_and_cache_features(DB_FILE, CONFIG.files.src_folder, more_scan=CONFIG.model.more_scan)

    # 3. Загружаем ВСЕ актуальные признаки из базы для сортировки
    LOGGER.info("Загружаем все признаки из базы данных для начала сортировки...")
    feats, paths = load_features_from_db(DB_FILE)

    if paths and feats is not None and len(paths) > 0:
        # Определим, будем ли использовать GPU для индекса FAISS
        use_gpu_faiss = (not CONFIG.model.use_cpu) and torch.cuda.is_available()

        # Если режим --find: Загружаем индекс с диска и ищем
        if CONFIG.search.find:

            use_gpu_faiss = (not CONFIG.model.use_cpu) and torch.cuda.is_available()

            # Загружаем индекс и mapping
            if not os.path.exists(CONFIG.files.index_file):
                LOGGER.error(f"! Индекс не найден: {CONFIG.files.index_file}. Сначала запустите без --find, чтобы его построить.")
                exit(2)

            index = load_faiss_index(CONFIG.files.index_file, use_gpu=use_gpu_faiss)
            order_map = np.load(CONFIG.files.index_file + ".order.npy")  # faiss_pos -> original_idx

            # Загрузим список путей и фич в original порядке
            feats, paths = load_features_from_db(DB_FILE)
            feats_f32 = feats.astype('float32', copy=False)

            # Словарь: basename -> list of original_idx (ускоряет поиск)
            name_to_idx = {}
            for i, p in enumerate(paths):
                base = os.path.basename(p)
                name_to_idx.setdefault(base, []).append(i)

            # --- функция поиска ---
            def find_neighbors_for_path(in_path: str):
                base = os.path.basename(in_path)
                candidates = name_to_idx.get(base)
                if not candidates:
                    return None, None, None  # не найден
                if len(candidates) > 1:
                    # несколько файлов с таким basename — ищем точное совпадение
                    matches = [i for i in candidates if paths[i] == in_path]
                    if len(matches) == 1:
                        orig_idx = matches[0]
                    else:
                        # неоднозначность
                        return "AMBIGUOUS", candidates, None
                else:
                    orig_idx = candidates[0]

                # ищем позицию в faiss (в final_order)
                pos_arr = np.where(order_map == orig_idx)[0]
                if pos_arr.size == 0:
                    return "NOT_IN_INDEX", orig_idx, None
                pos_in_faiss = int(pos_arr[0])

                k = max(1, CONFIG.search.find_NEIGHBORS)
                kk = min(k + 1, feats_f32.shape[0])
                D, I = index.search(feats_f32[orig_idx:orig_idx+1], kk)
                D = np.sqrt(D)
                neighbors = []
                for idx, dist in zip(I[0], D[0]):
                    if int(idx) == pos_in_faiss:
                        continue
                    neighbors.append(f"{int(idx)}:{float(dist):.6f}")
                    if len(neighbors) >= k:
                        break
                return orig_idx, pos_in_faiss, neighbors

            # --- Режим одиночного запроса ---
            if CONFIG.search.find != "__PIPE__":
                target = CONFIG.search.find
                orig_idx, pos, neighbors = find_neighbors_for_path(target)
                if orig_idx is None:
                    target = CONFIG.search.find

                # Сначала пытаемся найти в БД (как у вас уже реализовано)
                orig_idx, pos, neighbors = find_neighbors_for_path(target)
                                # --- Внешний файл (не найден в БД) — выводим 2 варианта ---
                if orig_idx is None:
                    target = CONFIG.search.find
                    LOGGER.info(f"'{target}' не найден в БД — пробуем как внешнюю картинку...")

                    # Проверки наличия сохранённых вспомогательных файлов
                    order_npy = CONFIG.files.index_file + ".order.npy"
                    paths_txt = CONFIG.files.index_file + ".paths.txt"
                    if (not os.path.exists(CONFIG.files.index_file)) or (not os.path.exists(paths_txt)) or (not os.path.exists(order_npy)):
                        LOGGER.error("! Для расширенного внешнего поиска требуется существующий faiss.index, .order.npy и .paths.txt (перестройте индекс).")
                        exit(2)

                    # Подгружаем список путей в порядке FAISS (faiss_pos -> path)
                    with open(paths_txt, "r", encoding="utf-8") as f:
                        faiss_pos_to_path = [line.strip() for line in f]

                    # Загружаем mapping faiss_pos -> original_idx
                    order_map = np.load(order_npy)

                    # Загружаем модель и извлекаем фичи для запроса
                    try:
                        model, hook = load_model(CONFIG.model.model_name)
                    except Exception as e:
                        print(f"! Не удалось загрузить модель для извлечения признаков: {e}")
                        exit(11)
                    try:
                        qfeat = extract_feature(target, model, hook, more_scan=CONFIG.model.more_scan)
                        if qfeat is None:
                            print(f"! Не удалось извлечь признаки для {target}")
                            exit(12)
                    except Exception as e:
                        print(f"! Ошибка при извлечении признаков для {target}: {e}")
                        exit(13)

                    q = qfeat.astype('float32').reshape(1, -1)

                    # 1) Глобальный поиск по FAISS — получаем позиции и расстояния
                    k = max(1, CONFIG.search.find_NEIGHBORS)
                    kk = min(k, max(1, index.ntotal))
                    try:
                        Dq, Iq = index.search(q, kk)
                        Dq = np.sqrt(Dq)
                    except Exception as e:
                        print(f"! Ошибка при поиске в FAISS: {e}")
                        exit(14)

                    # Если нет результатов — завершаем
                    if Iq.shape[1] == 0 or Iq.size == 0:
                        print(f"{target}\t-1\tNO_RESULTS")
                        exit(0)

                    # Найдём ближайшую позицию в faiss (первый элемент)
                    nearest_pos = int(Iq[0,0])
                    nearest_path = faiss_pos_to_path[nearest_pos]
                    nearest_orig_idx = int(order_map[nearest_pos])  # original index в paths

                    # 2) Вариант A: контекст — распечатаем соседей найденной в БД самой похожей картинки
                    # используем существующую функцию find_neighbors_for_path, чтобы получить её orig_idx,pos,neighbors
                    ctx_orig_idx, ctx_pos, ctx_neighbors = find_neighbors_for_path(nearest_path)
                    if ctx_orig_idx is None or ctx_orig_idx in ("AMBIGUOUS", "NOT_IN_INDEX"):
                        # На всякий случай — если по какой-то причине не нашли, печатаем простую строку
                        print(f"{nearest_path}\t{nearest_pos}\tCTX_NOT_FOUND")
                    else:
                        # печатаем именно так, как делает print_neighbors_indexed
                        print_neighbors_indexed(paths, ctx_orig_idx, ctx_pos, ctx_neighbors)

                    # 3) Вариант B: результаты прямого поиска по внешнему запросу (faiss_pos:dist)
                    neighbor_entries = []
                    for idx, dist in zip(Iq[0], Dq[0]):
                        idx = int(idx)
                        if idx < 0 or idx >= len(faiss_pos_to_path):
                            continue
                        neighbor_entries.append(f"{idx}:{float(dist):.6f}")
                        if len(neighbor_entries) >= k:
                            break

                    # Выводим глобальные соседи для внешней картинки (index = -1, поскольку её нет в БД)
                    print(f"{target}\t-1\t{','.join(neighbor_entries)}")

                    # Дополнительно: можно вызвать find_and_print_neighbors_simple для справки,
                    # но она ожидает, что image в БД; чтобы не ломать её контракт — не вызываем напрямую.
                    # Если вы хотите, чтобы её вывод тоже появлялся, раскомментируйте следующую строку:
                    # find_and_print_neighbors_simple(args.index_file, feats, paths, target, args.find_neighbors, use_gpu_faiss)

                    find_and_print_neighbors_simple_extended(
                        index=index,
                        faiss_pos_to_path=faiss_pos_to_path,
                        order_map=order_map,
                        feats=feats,
                        paths=paths,
                        query_path=target,
                        k=CONFIG.search.find_NEIGHBORS,
                        qvec=qfeat.astype('float32').reshape(1,-1),
                        extract_feature_fn=extract_feature,      # можно передать, но не используется, т.к. qvec передан
                        load_model_fn=load_model,                # тоже не нужен в этом вызове
                        model_name=CONFIG.model.model_name,
                        more_scan=CONFIG.model.more_scan
                    )

                    try:
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                    exit(0)
                elif orig_idx == "AMBIGUOUS":
                    print(f"{target}\t-1\tAMBIGUOUS")
                elif orig_idx == "NOT_IN_INDEX":
                    print(f"{target}\t-1\tNOT_INDEXED")
                else:
                    print_neighbors_indexed(paths,orig_idx,pos,neighbors)
                    find_and_print_neighbors_simple(CONFIG.files.index_file,feats,paths,target,CONFIG.search.find_NEIGHBORS,use_gpu_faiss)
                exit(0)

            # --- Режим pipeline ---
            # ------------------ Pipeline: чтение путей из stdin (замена) ------------------

            # Проверки наличия индекса и маппингов
            order_npy = CONFIG.files.index_file + ".order.npy"
            paths_txt = CONFIG.files.index_file + ".paths.txt"
            if (not os.path.exists(CONFIG.files.index_file)) or (not os.path.exists(paths_txt)) or (not os.path.exists(order_npy)):
                LOGGER.error("! Для pipeline требуется существующий faiss.index, .order.npy и .paths.txt (перестройте индекс).")
                exit(2)

            # Загрузка индекса и вспомогательных маппингов один раз
            index = load_faiss_index(CONFIG.files.index_file, use_gpu=use_gpu_faiss)
            with open(paths_txt, "r", encoding="utf-8") as f:
                faiss_pos_to_path = [line.strip() for line in f]
            order_map = np.load(order_npy)  # faiss_pos -> original_idx

            # Загрузим модель и hook один раз для всех внешних изображений
            try:
                model, hook = load_model(CONFIG.model.model_name)
            except Exception as e:
                LOGGER.error(f"! Не удалось загрузить модель для pipeline: {e}")
                exit(11)

            LOGGER.info("Pipeline mode started. Enter image paths (Ctrl+D/Ctrl+C to stop):", file=sys.stderr)

            # Основной цикл: читаем пути из stdin
            for raw in sys.stdin:
                line = raw.strip()
                if not line:
                    continue
                qpath = line

                # Сначала — пробуем найти в БД
                orig_idx, pos, neighbors = find_neighbors_for_path(qpath)

                if orig_idx is None:
                    # внешний — извлекаем признаки напрямую (переиспользуем модель)
                    try:
                        qfeat = extract_feature(qpath, model, hook, more_scan=CONFIG.model.more_scan)
                        if qfeat is None:
                            LOGGER.error(f"{qpath}\t-1\tERROR_EXTRACT")
                            continue
                    except Exception as e:
                        LOGGER.error(f"{qpath}\t-1\tERROR_EXTRACT:{e}")
                        continue

                    q = qfeat.astype('float32').reshape(1, -1)

                    # Выполняем поиск в FAISS
                    k = max(1, CONFIG.search.find_NEIGHBORS)
                    kk = min(k, max(1, index.ntotal))
                    try:
                        Dq, Iq = index.search(q, kk)
                        Dq = np.sqrt(Dq)
                    except Exception as e:
                        LOGGER.error(f"{qpath}\t-1\tERROR_FAISS:{e}")
                        continue

                    if Iq.size == 0:
                        LOGGER.error(f"{qpath}\t-1\tNO_RESULTS")
                        continue

                    # ближайшая позиция в faiss
                    nearest_pos = int(Iq[0, 0])
                    if nearest_pos < 0 or nearest_pos >= len(faiss_pos_to_path):
                        LOGGER.error(f"{qpath}\t-1\tNO_VALID_NEIGHBORS")
                        continue
                    nearest_path = faiss_pos_to_path[nearest_pos]
                    nearest_orig_idx = int(order_map[nearest_pos])

                    # Вариант A: контекст — печатаем соседей наиболее похожей DB-картинки
                    ctx_orig_idx, ctx_pos, ctx_neighbors = find_neighbors_for_path(nearest_path)
                    if ctx_orig_idx is None or ctx_orig_idx in ("AMBIGUOUS", "NOT_IN_INDEX"):
                        # Если по какой-то причине не нашли (маловероятно) — печатаем простую информацию
                        LOGGER.error(f"{nearest_path}\t{nearest_pos}\tCTX_NOT_FOUND")
                    else:
                        print_neighbors_indexed(paths, ctx_orig_idx, ctx_pos, ctx_neighbors)

                    # Вариант B: глобальные ближайшие к внешнему запросу (faiss_pos:dist)
                    neighbor_entries = []
                    for idx, dist in zip(Iq[0], Dq[0]):
                        idx = int(idx)
                        if idx < 0 or idx >= len(faiss_pos_to_path):
                            continue
                        neighbor_entries.append(f"{idx}:{float(dist):.6f}")
                        if len(neighbor_entries) >= k:
                            break
                    print(f"{qpath}\t-1\t{','.join(neighbor_entries)}")

                    find_and_print_neighbors_simple_extended(
                        index=index,
                        faiss_pos_to_path=faiss_pos_to_path,
                        order_map=order_map,
                        feats=feats,
                        paths=paths,
                        query_path=qpath,
                        k=CONFIG.search.find_NEIGHBORS,
                        qvec=qfeat.astype('float32').reshape(1,-1),
                        extract_feature_fn=extract_feature,      # можно передать, но не используется, т.к. qvec передан
                        load_model_fn=load_model,                # тоже не нужен в этом вызове
                        model_name=CONFIG.model.model_name,
                        more_scan=CONFIG.model.more_scan,
                    )

                elif orig_idx == "AMBIGUOUS":
                    LOGGER.error(f"{qpath}\t-1\tAMBIGUOUS")
                elif orig_idx == "NOT_IN_INDEX":
                    LOGGER.error(f"{qpath}\t-1\tNOT_INDEXED")
                else:
                    # найден в БД — печатаем оба варианта (как вы просили)
                    print_neighbors_indexed(paths, orig_idx, pos, neighbors)
                    find_and_print_neighbors_simple(CONFIG.files.index_file, feats, paths, qpath, CONFIG.search.find_NEIGHBORS, use_gpu_faiss)

            # Очистка модели / VRAM
            try:
                del model, hook
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            exit(0)
            # ------------------ Конец вставки ------------------


        # Если не режим find — строим (или перестраиваем) индекс и сохраняем его
        else:
            # стройка простого IndexFlatL2 и сохранение
            d = feats.shape[1]
            LOGGER.info(f"Построение FAISS индекса (IndexFlatL2) по {len(paths)} вектор(ов) размерности {d} ...")
            index = faiss.IndexFlatL2(d)
            if use_gpu_faiss:
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                except Exception as e:
                    LOGGER.info(f"  - Предупреждение: не удалось перенести индекс на GPU: {e}")
            index.add(feats.astype('float32', copy=False))

            # дальше — стандартная сортировка/поведение скрипта
            LOGGER.info(f"Всего в работе {len(paths)} изображений. Начинаем сортировку...")
            sort_images(feats, paths, CONFIG.files.dst_folder)