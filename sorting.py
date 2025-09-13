# ---------------------------------------------------------------------------- #
#                       Алгоритмы для поиска пути                              #
# ---------------------------------------------------------------------------- #

from collections import defaultdict
import time
import numpy as np
import torch
import tqdm

import faiss
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order, connected_components

from cli import ARG_BATCH_SIZE, ARG_INDEX_FILE, ARG_LIST_ONLY, ARG_LOG_LEVEL, ARG_LOOKAHEAD, ARG_NEIGHBORS_K_LIMIT, ARG_OUT_TSV, ARG_TSV_NEIGHBORS,ARG_TWO_OPT_BLOCK_SIZE,ARG_SORT_OPTIMIZER,ARG_TWO_OPT_SHIFT, ARG_USE_CPU
from faiss_io import save_faiss_index
from search import compute_global_knn
from utils import copy_and_rename

def get_adj_list(mst):
    adj = defaultdict(list)
    rows, cols = mst.nonzero()
    data = mst.data
    for r, c, dist in zip(rows, cols, data):
        adj[r].append((c, dist))
        adj[c].append((r, dist))
    return adj
        

def compute_greedy_walk_cost(start_node, adj, visited_in_main_dfs, depth):
    total_cost = 0.0
    current_node = start_node
    visited_in_walk = visited_in_main_dfs.copy()
    visited_in_walk.add(current_node)

    for _ in range(depth // 2):
        neighbors = [(nb, dist) for nb, dist in adj[current_node] if nb not in visited_in_walk]

        if not neighbors:
            break

        # Находим все возможные пары узлов и их суммарные стоимости
        min_pair_cost = float('inf')
        best_pair = None

        for nb1, dist1 in neighbors:
            # Находим соседей второго узла
            neighbors_of_nb1 = [(nb2, dist2) for nb2, dist2 in adj[nb1] if nb2 not in visited_in_walk and nb2 != current_node]
            for nb2, dist2 in neighbors_of_nb1:
                pair_cost = dist1 + dist2
                if pair_cost < min_pair_cost:
                    min_pair_cost = pair_cost
                    best_pair = (nb1, nb2, dist1, dist2)

        if best_pair is None:
            break

        next_node1, next_node2, dist1, dist2 = best_pair
        total_cost += dist1 + dist2

        # Обновляем текущий узел и добавляем посещенные узлы
        current_node = next_node2
        visited_in_walk.add(next_node1)
        visited_in_walk.add(next_node2)

    return total_cost


def optimized_depth_first_order(mst, i_start, lookahead_depth, progress_callback=None):
    n = mst.shape[0]
    adj = get_adj_list(mst)
    visited = np.zeros(n, dtype=bool)
    path = []
    stack = [i_start]
    cost_cache = {}
    
    while stack:
        node = stack.pop()
        if visited[node]:
            continue
        path.append(node)
        visited[node] = True
        
        if progress_callback:
            progress_callback(len(path))

        unvisited_neighbors = [nb for nb, _ in adj[node] if not visited[nb]]
        if unvisited_neighbors:
            visited_nodes_set = set(np.where(visited)[0])
            neighbor_costs = []
            for nb in unvisited_neighbors:
                if nb not in cost_cache:
                    cost_cache[nb] = compute_greedy_walk_cost(nb, adj, visited_nodes_set, depth=lookahead_depth)
                neighbor_costs.append((nb, cost_cache[nb]))
            
            sorted_neighbors = [nb for nb, cost in sorted(neighbor_costs, key=lambda x: x[1])]
            stack.extend(sorted_neighbors[::-1])
            
    if progress_callback:
        progress_callback(len(path), final_update=True)
            
    return np.array(path), visited


def sort_by_ann_mst(feats: np.ndarray, k: int, batch_size: int = ARG_BATCH_SIZE, use_gpu: bool = False, optimizer: str = ARG_SORT_OPTIMIZER, block_size: int = ARG_TWO_OPT_BLOCK_SIZE, shift: int = ARG_TWO_OPT_SHIFT):
    """
    Улучшенная сортировка на основе MST с обработкой несвязных графов ("островов").
    Использует ЕВКЛИДОВО (L2) РАССТОЯНИЕ.

    Args:
        feats (np.ndarray): Массив признаков (N, D).
        k (int): Количество соседей для поиска в k-NN графе.
        batch_size (int): Размер батча для поиска соседей.
        use_gpu (bool): Использовать ли GPU для FAISS.

    Returns:
        np.ndarray: Отсортированный порядок индексов (путь) или None в случае ошибки.
    """
    
    
    
    if faiss is None:
        if ARG_LOG_LEVEL == "default" or ARG_LOG_LEVEL == "error": print("! ОШИБКА: Библиотека 'faiss' не установлена. Сортировка невозможна.")
        return None

    n, d = feats.shape
    total_start_time = time.time()

    if ARG_LOG_LEVEL == "default":
        print("\n" + "="*80)
        print(f"Запуск улучшенной сортировки методом ANN + MST (на основе Евклидова расстояния)") # Изменено для ясности
        print(f"  - Изображений: {n}")
        print(f"  - Размерность фичей: {d}")
        print(f"  - Соседей на точку (k): {k}")
        print(f"  - Размер батча: {batch_size}")
        print(f"  - Использовать GPU: {use_gpu}")
        print(f"  - Оптимизатор: {optimizer} (block_size={block_size}, shift={shift})")
        print("="*80)

    # Работаем с копией float32 для FAISS
    feats_copy = feats.astype('float32').copy()

    # --- Шаг 1: Индексация в FAISS ---
    step_start_time = time.time()
    if ARG_LOG_LEVEL == "default": print(f"\n[1/5] Шаг 1: Индексация векторов в FAISS...")
    try:
        # ИЗМЕНЕНИЕ 1: Убираем нормализацию. Она не нужна для евклидова расстояния.
        # faiss.normalize_L2(feats_copy)

        # ИЗМЕНЕНИЕ 2: Используем IndexFlatL2 для евклидова расстояния вместо IndexFlatIP.
        index = faiss.IndexFlatL2(d)

        if use_gpu and torch is not None and torch.cuda.is_available():
            if ARG_LOG_LEVEL == "default": print("  - Попытка использовать GPU для FAISS...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            if ARG_LOG_LEVEL == "default": print("  - Индекс успешно перенесен на GPU.")

        index.add(feats_copy)
        if ARG_LOG_LEVEL == "default": print(f"  - Индекс создан. Всего векторов: {index.ntotal}.")
        if ARG_LOG_LEVEL == "default": print(f"  - Время на шаг 1: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        if ARG_LOG_LEVEL == "default" or ARG_LOG_LEVEL == "error": print(f"! Ошибка на шаге 1: {e}")
        return None

    # --- Шаг 2: Поиск k-ближайших соседей ---
    step_start_time = time.time()
    if ARG_LOG_LEVEL == "default": print(f"\n[2/5] Шаг 2: Поиск {k} ближайших соседей...")
    try:
        all_distances = []
        all_indices = []
        for i in tqdm(range(0, n, batch_size), desc="  - Поиск k-NN (батчи)"):
            end = min(i + batch_size, n)
            # Ищем k+1 соседа, так как первый результат - это сама точка
            distances_batch, indices_batch = index.search(feats_copy[i:end], k + 1)
            all_distances.append(distances_batch)
            all_indices.append(indices_batch)
        
        # Для IndexFlatL2 возвращаются КВАДРАТЫ евклидовых расстояний.
        # Переименуем для ясности.
        distances_sq = np.vstack(all_distances)
        indices = np.vstack(all_indices)
        if ARG_LOG_LEVEL == "default": print(f"  - Поиск завершен. Размер матрицы индексов: {indices.shape}")
        if ARG_LOG_LEVEL == "default": print(f"  - Время на шаг 2: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        if ARG_LOG_LEVEL == "default" or ARG_LOG_LEVEL == "error": print(f"! Ошибка на шаге 2: {e}")
        return None

    # --- Шаг 3: Построение симметричного разреженного графа ---
    step_start_time = time.time()
    if ARG_LOG_LEVEL == "default": print(f"\n[3/5] Шаг 3: Построение симметричного графа...")
    try:
        # Индексы соседей (исключаем первый столбец, т.к. это сама точка)
        cols = indices[:, 1:].flatten()
        # Индексы исходных точек, повторенные k раз
        rows = np.arange(n).repeat(k)

        # ИЗМЕНЕНИЕ 3: Расчет стоимости.
        # Квадрат расстояния уже является стоимостью (чем меньше, тем лучше).
        # Нам не нужно преобразовывать его из сходства.
        costs = distances_sq[:, 1:].flatten()
        # Проверка на отрицательные значения (хотя для L2 они маловероятны) не помешает.
        costs[costs < 0] = 0

        # Создаем асимметричный граф
        asymmetric_graph = csr_matrix((costs, (rows, cols)), shape=(n, n))

        # Делаем граф симметричным, выбирая минимальную стоимость ребра
        symmetric_graph = asymmetric_graph.minimum(asymmetric_graph.T)

        if ARG_LOG_LEVEL == "default": print(f"  - Граф успешно создан. Количество ребер: {symmetric_graph.nnz}.")
        if ARG_LOG_LEVEL == "default": print(f"  - Время на шаг 3: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        if ARG_LOG_LEVEL == "default" or ARG_LOG_LEVEL == "error": print(f"! Ошибка на шаге 3: {e}")
        return None

    # --- Шаги 4 и 5 остаются без изменений, так как они работают с графом,
    # которому неважно, как были получены веса ребер. ---

    # --- Шаг 4: Построение Minimum Spanning Tree (MST) ---
    step_start_time = time.time()
    if ARG_LOG_LEVEL == "default": print(f"\n[4/5] Шаг 4: Построение Minimum Spanning Tree...")
    try:
        mst = minimum_spanning_tree(symmetric_graph)
        if ARG_LOG_LEVEL == "default": print(f"  - MST построено. Общая стоимость дерева: {mst.sum():.2f}.")
        if ARG_LOG_LEVEL == "default": print(f"  - Время на шаг 4: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        if ARG_LOG_LEVEL == "default" or ARG_LOG_LEVEL == "error": print(f"! Ошибка на шаге 4: {e}")
        return None

    # --- Шаг 5


    # --- Начало исправленного блока для Шага 5 ---
    step_start_time = time.time()
    if ARG_LOG_LEVEL == "default": print(f"\n[5/5] Шаг 5: Двухэтапная кластеризация и обход...")
    
    try:
        n = mst.shape[0]
        # Определяем все связанные компоненты в исходном MST
        n_components, labels = connected_components(csgraph=mst, directed=False, return_labels=True)
        
        # --- ЭТАП 1: ОБРАБОТКА ОСНОВНОЙ КОМПОНЕНТЫ ---
        if ARG_LOG_LEVEL == "default": print(f"\n  --- Этап 1: Обработка основной компоненты ---")
        if n_components > 0:
            component_sizes = np.bincount(labels)
            main_component_id = np.argmax(component_sizes)
            main_nodes_mask = (labels == main_component_id)
            main_nodes_indices = np.where(main_nodes_mask)[0]
            total_in_main_component = len(main_nodes_indices)
            
            if ARG_LOG_LEVEL == "default": print(f"  - Найдена основная компонента размером {total_in_main_component} узлов.")
            
            start_node = main_nodes_indices[0]
            
            LOOKAHEAD_DEPTH = ARG_LOOKAHEAD # Можете менять это значение, оно почти ни на что не влияет так как график строится жадно без ветвлений (например, 30, 50, 100, 0/1=выкл)
            
            # --- ГЛАВНЫЙ ПЕРЕКЛЮЧАТЕЛЬ АЛГОРИТМОВ ---
            if LOOKAHEAD_DEPTH <= 1:
                if ARG_LOG_LEVEL == "default": print(f"  - Глубина просмотра (depth={LOOKAHEAD_DEPTH}) <= 1. Используется стандартный быстрый DFS.")
                from scipy.sparse.csgraph import depth_first_order
                # Запускаем простой DFS на всем MST, начиная с узла из главной компоненты.
                # Он обойдет только свою компоненту связности.
                main_path_indices, _ = depth_first_order(mst, i_start=start_node, directed=False)
                main_path = main_path_indices[main_path_indices != -1] # Убираем непосещенные узлы
                if ARG_LOG_LEVEL == "default": print(f"    - Стандартный DFS завершен.")
            else:
                if ARG_LOG_LEVEL == "default": print(f"  - Глубина просмотра (depth={LOOKAHEAD_DEPTH}). Используется DFS с lookahead-оптимизацией.")
                last_reported_percent = -1
                def report_progress(processed_count, final_update=False):
                    nonlocal last_reported_percent
                    percent_done = int((processed_count / total_in_main_component) * 100)
                    if percent_done > last_reported_percent or final_update:
                        if ARG_LOG_LEVEL == "default": print(f"    - Прогресс: {processed_count} / {total_in_main_component} узлов ({percent_done}%)", end='\r')
                        last_reported_percent = percent_done
                
                main_path, _ = optimized_depth_first_order(
                    mst, 
                    start_node, 
                    lookahead_depth=LOOKAHEAD_DEPTH,
                    progress_callback=report_progress
                )
                if ARG_LOG_LEVEL == "default": print() # Перенос строки после прогресс-бара
            
            if ARG_LOG_LEVEL == "default": print(f"  - Основная компонента обработана. Длина основного пути: {len(main_path)}.")
            
        else:
            # Редкий случай, если граф пуст
            if ARG_LOG_LEVEL == "default": print("  - Не найдено ни одной компоненты. Основной путь пуст.")
            main_path = np.array([], dtype=int)
            main_nodes_mask = np.zeros(n, dtype=bool)
    
        # --- ЭТАП 2: ОБРАБОТКА ОСТАВШИХСЯ "ОСТРОВОВ" ---
        if ARG_LOG_LEVEL == "default": print(f"\n  --- Этап 2: Обработка оставшихся компонент ('островов') ---")
        
        island_nodes_indices = np.where(~main_nodes_mask)[0]
        num_islands = len(island_nodes_indices)
        
        secondary_path = np.array([], dtype=int) # Инициализируем вторичный путь
    
        if num_islands > 1:
            if ARG_LOG_LEVEL == "default": print(f"  - Найдено {num_islands} узлов в 'островах'. Запускаем для них отдельный процесс ANN+MST+DFS.")
            
            # Шаг 2.1: Извлекаем фичи только для "островов"
            island_feats = feats_copy[island_nodes_indices]
            
            # Шаг 2.2: ANN. Строим граф "все ко всем" (k = N-1)
            if ARG_LOG_LEVEL == "default": print(f"    - [2.1/2.4] Поиск соседей для {num_islands}...")
            island_k = min(1000, num_islands - 1) # k не может быть больше N-1
            island_index = faiss.IndexFlatL2(island_feats.shape[1])
            if use_gpu: # Предполагается, что use_gpu определена ранее
                res = faiss.StandardGpuResources()
                island_index = faiss.index_cpu_to_gpu(res, 0, island_index)
            island_index.add(island_feats)
            island_dists, island_neighbors_local_idx = island_index.search(island_feats, k=island_k)
            
            # Шаг 2.3: Построение графа и MST для "островов"
            if ARG_LOG_LEVEL == "default": print(f"    - [2.2/2.4] Построение графа для 'островов'...")
            
            # Индексы из FAISS - локальные (от 0 до num_islands-1).
            rows = np.arange(num_islands).repeat(island_k)
            cols = island_neighbors_local_idx.flatten()
            data = island_dists.flatten()
            
            # Убираем петли (i,i) и некорректные расстояния
            valid_mask = (rows != cols) & (data > 0)
            island_graph = coo_matrix((data[valid_mask], (rows[valid_mask], cols[valid_mask])), shape=(num_islands, num_islands))
            island_graph.eliminate_zeros()
            
            if ARG_LOG_LEVEL == "default": print(f"    - [2.3/2.4] Построение MST для 'островов'...")
            island_mst = minimum_spanning_tree(island_graph)
            
            # Шаг 2.4: Простой DFS для MST "островов"
            if ARG_LOG_LEVEL == "default": print(f"    - [2.4/2.4] Запуск простого DFS для 'островов'...")
            from scipy.sparse.csgraph import depth_first_order
            
            # Начинаем с первого узла в локальном списке "островов"
            island_path_local, _ = depth_first_order(island_mst, i_start=0, directed=False)
            island_path_local_visited = island_path_local[island_path_local != -1]
            
            # Преобразуем локальные индексы пути (0..num_islands-1) в глобальные индексы изображений
            secondary_path = island_nodes_indices[island_path_local_visited]
            if ARG_LOG_LEVEL == "default":  print(f"  - 'Острова' обработаны. Длина вторичного пути: {len(secondary_path)}.")
            
        elif num_islands == 1:
            if ARG_LOG_LEVEL == "default": print("  - Найден 1 узел в 'островах', добавляем его напрямую.")
            secondary_path = island_nodes_indices
        else:
            if ARG_LOG_LEVEL == "default": print("  - Нет 'островов' для обработки.")
    
        # --- ЭТАП 3: ФИНАЛЬНАЯ СБОРКА И ПРОВЕРКА ---
        if ARG_LOG_LEVEL == "default": print(f"\n  --- Этап 3: Финальная сборка пути ---")
        
        # Собираем основной путь и путь "островов"
        path_list = list(main_path) + list(secondary_path)
        
        # Проверяем, все ли узлы были посещены. Добавляем "потерянные" в конец.
        if len(path_list) != n:
            if ARG_LOG_LEVEL == "default" or ARG_LOG_LEVEL == "error": print(f"  - ! ПРЕДУПРЕЖДЕНИЕ: Длина пути ({len(path_list)}) не равна общему числу элементов ({n}).")
            all_nodes = set(range(n))
            path_nodes = set(path_list)
            unvisited = list(all_nodes - path_nodes)
            if ARG_LOG_LEVEL == "default": print(f"    - Найдено {len(unvisited)} непосещенных узлов. Добавляем их в конец.")
            path_list.extend(unvisited)
        
        path = np.array(path_list)
        
        if ARG_LOG_LEVEL == "default": print(f"  - Сборка завершена. Получен полный путь из {len(path)} элементов.")
        if ARG_LOG_LEVEL == "default": print(f"  - Время на шаг 5: {time.time() - step_start_time:.2f} сек.")
    
    except Exception as e:
        if ARG_LOG_LEVEL == "default" or ARG_LOG_LEVEL == "error": print(f"\n! КРИТИЧЕСКАЯ ОШИБКА на шаге 5: {e}")
        import traceback
        traceback.print_exc()
        path = None # Возвращаем None в случае ошибки
    
   
    # --- Шаг 6: Пост-обработка пути с оптимизатором ---
    step_start_time = time.time()
    if ARG_LOG_LEVEL == "default": print(f"\n[6/6] Шаг 6: Пост-обработка пути с {optimizer} (блоки {block_size}, сдвиг {shift})...")
    
    # Вспомогательные функции для оптимизаторов
    def compute_distance_matrix(sub_feats):
        """Плотная матрица L2 расстояний для блока."""
        diff = sub_feats[:, None] - sub_feats[None, :]
        return np.sqrt(np.sum(diff**2, axis=-1))
    
    def two_opt(subpath, sub_feats):
        """Простая 2-opt оптимизация для Hamiltonian path с фиксированными концами."""
        n_sub = len(subpath)
        if n_sub < 4: # Для 2-opt с фикс. концами нужно хотя бы 4 точки
            return subpath
        
        dist_matrix = compute_distance_matrix(sub_feats)
        path_indices = np.arange(n_sub) # Работаем с индексами 0..n-1
        
        improved = True
        while improved:
            improved = False
            for i in range(1, n_sub - 2):
                for j in range(i + 1, n_sub - 1):
                    # Старые ребра: (i-1 -> i) и (j -> j+1)
                    old_dist = dist_matrix[path_indices[i-1], path_indices[i]] + dist_matrix[path_indices[j], path_indices[j+1]]
                    # Новые ребра: (i-1 -> j) и (i -> j+1)
                    new_dist = dist_matrix[path_indices[i-1], path_indices[j]] + dist_matrix[path_indices[i], path_indices[j+1]]
                    
                    if new_dist < old_dist - 1e-6:
                        path_indices[i:j+1] = path_indices[i:j+1][::-1]
                        improved = True
        return subpath[path_indices]

    
    # Разбиение на overlapping блоки и оптимизация
    optimized_path = path.copy()
    num_blocks = max(1, (n - block_size) // shift + 1)
    
    for b in tqdm(range(num_blocks), desc="  - Оптимизация блоков"):
        start = b * shift
        end = min(start + block_size, n)
        if end - start < 3:
            continue
        
        subpath = optimized_path[start:end]
        sub_feats = feats[subpath]
        
        # В вашей логике концы всегда фиксированы, кроме, возможно, крайних блоков.
        # Для простоты здесь всегда считаем их фиксированными, т.к. они соединяются с остальной частью пути.
        # Если нужна особая логика для крайних блоков, ее можно добавить сюда.
        fixed_ends = True
        
        if optimizer == '2opt':
            optimized_path[start:end] = two_opt(subpath, sub_feats, fixed_ends=fixed_ends)

    
    path = optimized_path
    if ARG_LOG_LEVEL == "default": print(f"  - Пост-обработка завершена. Время: {time.time() - step_start_time:.2f} сек.")

    
    # --- Финальный вывод (замените ваши оригинальные print'ы на это, чтобы учесть шаг 6) ---
    if ARG_LOG_LEVEL == "default":
        print("\n" + "="*80)
        print("Сортировка методом ANN + MST успешно завершена.")
        print(f"Общее время выполнения: {time.time() - total_start_time:.2f} сек.")
        print("="*80 + "\n")
    
    return path

def sort_images(feats, paths, out_folder):
    n = feats.shape[0]
    if n < 1:
        if ARG_LOG_LEVEL == "default": print("Нет изображений для сортировки.")
        return

    if ARG_LOG_LEVEL == "default":
        print(f"\n# {'-'*76} #")
        print(f"# Сортировка изображений методом 'ANN+MST'") 
        print(f"# {'-'*76} #")
        print(f"Всего изображений: {n}")

    # Подбираем разумное k для внутреннего графа (как раньше)
    if len(paths) < ARG_NEIGHBORS_K_LIMIT:
        k_neighbors = len(paths) - 1
    else:
        k_neighbors = ARG_NEIGHBORS_K_LIMIT

    use_gpu_faiss = not ARG_USE_CPU and torch.cuda.is_available()
    final_order = sort_by_ann_mst(feats, k=k_neighbors, use_gpu=use_gpu_faiss)

    if final_order is None or len(final_order) == 0:
        if ARG_LOG_LEVEL == "default" or ARG_LOG_LEVEL == "error": print("! Сортировка не удалась, итоговый путь пуст. Операция отменена.")
        return
    
    # предположим: feats: np.ndarray shape (N,D), final_order: np.ndarray length N (original indices)
    ordered_indices = np.array(final_order, dtype=np.int64)
    ordered_feats = feats[ordered_indices]   # теперь порядок соответствует final_order
    
    # создаём индекс и добавляем ordered_feats
    d = ordered_feats.shape[1]
    index = faiss.IndexFlatL2(d)
    # (опционально GPU перенос)
    index.add(ordered_feats.astype('float32', copy=False))
    
    # сохраняем индекс
    save_faiss_index(index,ARG_INDEX_FILE)
    
    # сохраняем mapping (faiss position -> original index)
    np.save(ARG_INDEX_FILE + ".order.npy", ordered_indices)   # файл: faiss.index.order.npy
    
    # (опционально) сохраним также список путей в том же порядке — удобно для отладки
    ordered_paths = [paths[i] for i in ordered_indices]
    with open(ARG_INDEX_FILE + ".paths.txt", "w", encoding="utf-8") as f:
        for p in ordered_paths:
            f.write(p + "\n")


    # Если пользователь запросил только список — формируем TSV с глобальными соседями
    if ARG_LIST_ONLY:
        # Вычисляем глобальные k-NN (по всей базе). Используем args.neighbors
        knn_k = max(1, ARG_TSV_NEIGHBORS)
        if ARG_LOG_LEVEL == "default": print(f"\nВычисляем глобальные {knn_k} ближайших соседей и формируем файл...")

        # Сначала постройте таблицу обратной индексации: original_index -> position in final order
        inverse_order = np.empty(len(final_order), dtype=int)
        for pos, orig_idx in enumerate(final_order):
            inverse_order[orig_idx] = pos

        # Выполняем глобальный поиск соседей (возвращаются индексы по оригинальным индексам)
        knn_idxs, knn_dists = compute_global_knn(feats, knn_k, batch_size=ARG_BATCH_SIZE, use_gpu=use_gpu_faiss)

        out_file = ARG_OUT_TSV
        # Записываем в TSV потоково
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("path\tindex\tdistance\n")
            # Проходим по позициям в итоговом порядке (index — позиция в упорядоченной последовательности)
            for pos_in_order, orig_idx in enumerate(final_order):
                path = paths[orig_idx]
                neighbor_entries = []
                # Берём для этой точки её глобальных соседей (в порядке от ближнего к дальнему)
                neighs = knn_idxs[orig_idx]
                dists = knn_dists[orig_idx]
                for ni, dist in zip(neighs, dists):
                    if ni == -1:
                        continue
                    # переводим оригинальный индекс соседа в индекс в итоговом порядке
                    neighbor_pos = int(inverse_order[ni])
                    neighbor_entries.append(f"{neighbor_pos}:{dist:.6f}")
                line = f"{path}\t{pos_in_order}\t{','.join(neighbor_entries)}\n"
                f.write(line)

        # В консоль выводим только путь к файлу
        print(out_file)
        return

    # Если list_only не указан — поведение прежнее: копируем файлы в out_folder
    copy_and_rename(paths, final_order, out_folder)
    if ARG_LOG_LEVEL == "default": print("Сортировка завершена.")
