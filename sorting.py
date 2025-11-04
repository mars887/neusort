# ---------------------------------------------------------------------------- #
#                       Алгоритмы для поиска пути                              #
# ---------------------------------------------------------------------------- #

import os
import time
import numpy as np
import torch
from tqdm.auto import tqdm

import faiss
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

from config import Config
from faiss_io import save_faiss_index
from cli import LOGGER
from metrics import evaluate_order_metrics
from search import compute_global_knn
from utils import copy_and_rename, to_local_order


import numpy as np
from typing import Optional, Tuple, List
from tqdm.auto import tqdm
from config import Config

def farthest_insertion_path(
    feats: np.ndarray,
    config: Config,
    *,
    window: int = 40,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Быстрый Farthest Insertion:
    - two-sweep старт (O(N)+O(N)), вместо O(N^2) перебора пары;
    - вставка только в окрестности ближайшего узла пути (±window рёбер) + концы;
    - векторизованное обновление кэша расстояний (квадраты), хранение ближайшего узла;
    - путь хранится как двусвязный список (вставка O(1)).

    Аргументы:
      feats: (N, D) float32/float64.
      config: Config (для совместимости).
      window: радиус окна по числу рёбер вокруг опорного узла (реком. 24–48 для кластеров ~300–1000).
      seed: детерминированность two-sweep.

    Возвращает:
      order: np.ndarray длиной N — порядок обхода.
    """
    X = feats.astype(np.float32, copy=False)
    N, D = X.shape
    if N == 0:
        return np.empty((0,), dtype=np.int64)
    if N == 1:
        return np.array([0], dtype=np.int64)
    if window <= 0:
        window = 1

    rng = np.random.default_rng(seed if seed is not None else 42)

    # --- two-sweep старт: r -> a (max dist), a -> b (max dist) ---
    r = int(rng.integers(0, N))
    d_r = np.einsum('ij,ij->i', X - X[r], X - X[r])  # ||x - r||^2
    a = int(np.argmax(d_r))
    d_a = np.einsum('ij,ij->i', X - X[a], X - X[a])
    b = int(np.argmax(d_a))
    if a == b:
        b = (a + 1) % N

    # двусвязный список пути
    prev = np.full(N, -1, dtype=np.int64)
    next_ = np.full(N, -1, dtype=np.int64)
    head, tail = a, b
    prev[b] = a
    next_[a] = b

    visited = np.zeros(N, dtype=bool)
    visited[a] = True
    visited[b] = True

    # кэш: min расстояние до пути и "кто ближайший"
    da2 = np.einsum('ij,ij->i', X - X[a], X - X[a])
    db2 = np.einsum('ij,ij->i', X - X[b], X - X[b])
    dist_cache_sq = np.minimum(da2, db2)
    nearest_node = np.where(da2 <= db2, a, b)

    # посещённые не должны выбираться снова
    dist_cache_sq[a] = -np.inf
    dist_cache_sq[b] = -np.inf

    def walk_left(u: int, steps: int) -> List[int]:
        seq = []
        cur = u
        for _ in range(steps):
            p = prev[cur]
            if p == -1:
                break
            seq.append(p)
            cur = p
        seq.reverse()
        return seq

    def walk_right(u: int, steps: int) -> List[int]:
        seq = []
        cur = u
        for _ in range(steps):
            nx = next_[cur]
            if nx == -1:
                break
            seq.append(nx)
            cur = nx
        return seq

    def gather_candidate_edges(anchor: int, w: int) -> List[Tuple[Optional[int], Optional[int]]]:
        # кандидаты: (None, head), рёбра в окрестности anchor, (tail, None)
        left_nodes = walk_left(anchor, w) + [anchor]
        right_nodes = walk_right(anchor, w)
        nodes_span = left_nodes + right_nodes

        edges: List[Tuple[Optional[int], Optional[int]]] = []
        if head != -1:
            edges.append((None, head))
        for i in range(len(nodes_span) - 1):
            u, v = nodes_span[i], nodes_span[i + 1]
            if u != -1 and v != -1 and next_[u] == v:
                edges.append((u, v))
        if tail != -1:
            edges.append((tail, None))

        seen = set()
        uniq = []
        for e in edges:
            if e not in seen:
                uniq.append(e)
                seen.add(e)
        return uniq

    def best_insertion_for_point(f: int, anchor: int, w: int) -> Tuple[Optional[int], Optional[int], float]:
        cand_edges = gather_candidate_edges(anchor, w)
        fvec = X[f]
        u_list = [u for (u, v) in cand_edges]
        v_list = [v for (u, v) in cand_edges]

        inc = np.empty(len(cand_edges), dtype=np.float32)

        mask_u = np.array([u is not None for u in u_list])
        mask_v = np.array([v is not None for v in v_list])
        u_idx = np.array([u if u is not None else 0 for u in u_list], dtype=np.int64)
        v_idx = np.array([v if v is not None else 0 for v in v_list], dtype=np.int64)

        inc.fill(np.inf)

        both = mask_u & mask_v
        if both.any():
            uu = u_idx[both]
            vv = v_idx[both]
            inc[both] = (
                np.linalg.norm(X[uu] - fvec, axis=1)
                + np.linalg.norm(X[vv] - fvec, axis=1)
                - np.linalg.norm(X[uu] - X[vv], axis=1)
            )

        left_end = (~mask_u) & mask_v  # (None, head)
        if left_end.any():
            vv = v_idx[left_end]
            inc[left_end] = np.linalg.norm(X[vv] - fvec, axis=1)

        right_end = mask_u & (~mask_v)  # (tail, None)
        if right_end.any():
            uu = u_idx[right_end]
            inc[right_end] = np.linalg.norm(X[uu] - fvec, axis=1)

        k = int(np.argmin(inc))
        return (u_list[k], v_list[k], float(inc[k]))

    remaining = int(N - 2)
    pbar = tqdm(total=remaining, desc="  - Farthest Insertion (fast)", leave=False)

    while remaining > 0:
        f = int(np.argmax(dist_cache_sq))      # самая дальняя точка (посещённые = -inf)
        anchor = int(nearest_node[f])          # ближайший узел пути для f (из кэша)

        u, v, _ = best_insertion_for_point(f, anchor, window)

        # вставка в список
        if u is None and v is not None:        # перед head
            prev[v] = f
            next_[f] = v
            prev[f] = -1
            head = f
        elif v is None and u is not None:      # после tail
            next_[u] = f
            prev[f] = u
            next_[f] = -1
            tail = f
        else:                                  # между u и v
            next_[u] = f
            prev[f] = u
            next_[f] = v
            prev[v] = f

        # обновление кэшей
        visited[f] = True
        dist_cache_sq[f] = -np.inf

        mask = ~visited
        if mask.any():
            dif = X[mask] - X[f]
            d2 = np.einsum('ij,ij->i', dif, dif)
            better = d2 < dist_cache_sq[mask]
            dist_cache_sq[mask] = np.where(better, d2, dist_cache_sq[mask])
            tmp = nearest_node[mask]
            tmp[better] = f
            nearest_node[mask] = tmp

        remaining -= 1
        pbar.update(1)

    pbar.close()

    # восстановление порядка из двусвязного списка
    order = np.empty(N, dtype=np.int64)
    cur, i = head, 0
    while cur != -1:
        order[i] = cur
        cur = next_[cur]
        i += 1

    # страховка на случай рассинхронизации (не должна сработать)
    if i < N:
        missing = np.setdiff1d(np.arange(N, dtype=np.int64), order[:i], assume_unique=False)
        order[i:i+len(missing)] = missing

    return order


from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss
import math

def farthest_insertion_path_clustered(
    feats: np.ndarray,
    config,
    n_clusters: int = None,
    parallel: bool = True,
) -> np.ndarray:
    """
    Clustered Farthest Insertion (качество + стабильность, с расширенным логированием).

    Этапы:
      1) K по max_cluster_size; Faiss KMeans
      2) Локальные пути: farthest_insertion_path (окно ~6% размера кластера)
         2.1) Intra-cluster refine: ограниченный 2-opt в «проблемных» кластерах
      3) Порядок кластеров: MST(centroids)+DFS → centroid 2-opt → соседний swap
      4) Склейка (list-based): append + adaptive bridge-rotation + lookahead по W кластерам
         (фиксируем границы блоков для последующего boundary-refine)
      5) Boundary-refine: микро 2-opt на каждой границе (окно 2L)
      6) Polish (по массиву порядка): локальная 2-opt по окнам
      7) Integrity-check, телеметрия: процентили длин рёбер, top jumps, top inter-cluster jumps

    Конфиг (опционально):
      sorting.max_cluster_size        (int, дефолт 450)
      sorting.bridge_rotate_r         (int, дефолт 32)
      sorting.merge_lookahead         (int, дефолт 10)
      sorting.polish_window           (int, дефолт 136)
      sorting.polish_passes           (int, дефолт 3)
      sorting.intra_cluster_refine    (bool, дефолт True)
      sorting.intra_cluster_refine_threshold (float, дефолт 1.15)
      sorting.intra_cluster_refine_max_swaps (int, дефолт 400)
      sorting.boundary_refine_L       (int, дефолт 12)
    """
    t_all0 = time.perf_counter()
    X = feats.astype(np.float32, copy=False)
    N, D = X.shape
    if N == 0:
        return np.array([], dtype=np.int64)
    if N == 1:
        return np.array([0], dtype=np.int64)
    LOGGER.info(f"  - Clustered FI: N={N}, D={D}, parallel={parallel}")

    # ------------------ параметры из конфига ------------------
    max_cluster_size = int(getattr(config.sorting, "max_cluster_size", 450))
    bridge_rotate_r  = int(getattr(config.sorting, "bridge_rotate_r", 32))
    merge_lookahead  = int(getattr(config.sorting, "merge_lookahead", 10))
    polish_window    = int(getattr(config.sorting, "polish_window", 136))
    polish_passes    = int(getattr(config.sorting, "polish_passes", 3))

    intra_refine_enabled   = bool(getattr(config.sorting, "intra_cluster_refine", True))
    intra_refine_thr       = float(getattr(config.sorting, "intra_cluster_refine_threshold", 1.15))
    intra_refine_max_swaps = int(getattr(config.sorting, "intra_cluster_refine_max_swaps", 400))

    boundary_refine_L      = int(getattr(config.sorting, "boundary_refine_L", 12))

    # ------------------ выбор K ------------------
    if n_clusters is None:
        n_clusters = int(math.ceil(N / max(1, max_cluster_size)))
        n_clusters = max(32, min(n_clusters, max(32, N // 6)))
    n_clusters = min(n_clusters, N)
    LOGGER.info(f"    - K selection: K={n_clusters} (max_cluster_size≈{int(math.ceil(N/max(1,n_clusters)))}).")

    # ------------------ KMeans (faiss) ------------------
    t0 = time.perf_counter()
    km = faiss.Kmeans(d=D, k=n_clusters, niter=25, verbose=False); km.nredo = 3
    km.train(X)
    centroids0 = km.centroids.astype(np.float32)
    cindex = faiss.IndexFlatL2(D); cindex.add(centroids0)
    _, labels = cindex.search(X, 1); labels = labels.reshape(-1)

    clusters: List[np.ndarray] = []
    for c in range(n_clusters):
        inds = np.where(labels == c)[0]
        if inds.size > 0:
            clusters.append(inds.astype(np.int64))
    n_clusters = len(clusters)
    t1 = time.perf_counter()
    LOGGER.info(f"    - KMeans done: non-empty K={n_clusters} in {t1 - t0:.2f}s")

    centroids = np.vstack([X[idxs].mean(axis=0) for idxs in clusters]).astype(np.float32)
    sizes = np.array([len(idxs) for idxs in clusters], dtype=np.int64)
    LOGGER.info(f"    - cluster sizes: min={sizes.min()}, p25={np.percentile(sizes,25):.0f}, p50={np.median(sizes):.0f}, "
                f"p75={np.percentile(sizes,75):.0f}, p90={np.percentile(sizes,90):.0f}, max={sizes.max()}")

    # Узел->кластер (для диагностики межкластерных прыжков)
    node2cluster = np.full(N, -1, dtype=np.int32)

    # ------------------ локальные пути + intra-cluster refine ------------------
    def _edge_stats(seq: np.ndarray) -> Tuple[float, float, float, float]:
        if seq.size < 2:
            return 0.0, 0.0, 0.0, 0.0
        P = X[seq]
        dist = np.linalg.norm(P[1:] - P[:-1], axis=1)
        return float(dist.mean()), float(np.percentile(dist, 90)), float(dist.max()), float(np.median(dist))

    def _refine_cluster_2opt(seq: np.ndarray, max_swaps: int = 200, step_a: int = 6, step_b: int = 6) -> np.ndarray:
        """Узкая 2-opt внутри кластера: лимит по числу «выгодных» разворотов."""
        L = int(seq.size)
        if L < 16 or max_swaps <= 0:
            return seq
        swaps = 0
        improved = True
        while improved and swaps < max_swaps:
            improved = False
            # один проход с разреженным шагом
            for a in range(1, L - 2, step_a):
                if swaps >= max_swaps: break
                for b in range(a + 10, L - 1, step_b):
                    u1, v1 = seq[a - 1], seq[a]
                    u2, v2 = seq[b], (seq[b + 1] if b + 1 < L else None)
                    before = np.linalg.norm(X[u1] - X[v1]) + (np.linalg.norm(X[u2] - X[v2]) if v2 is not None else 0.0)
                    after  = np.linalg.norm(X[u1] - X[u2]) + (np.linalg.norm(X[v1] - X[v2]) if v2 is not None else 0.0)
                    if after + 1e-9 < before:
                        seq[a:b + 1] = seq[a:b + 1][::-1]
                        swaps += 1
                        improved = True
                        if swaps >= max_swaps:
                            break
        return seq

    def _process_cluster(ci: int) -> Tuple[int, np.ndarray, np.ndarray, Tuple[float,float,float,float]]:
        inds = clusters[ci]
        local = X[inds]
        # окно ~6% (48..128)
        w = max(48, min(128, int(0.06 * len(inds))))
        local_order_local = farthest_insertion_path(local, config, window=w)
        seq = inds[local_order_local]

        # исходная статистика
        mean_e, p90_e, max_e, med_e = _edge_stats(seq)

        # intra-cluster refine при необходимости
        did_refine = False
        if intra_refine_enabled and (max_e > intra_refine_thr or p90_e > intra_refine_thr * 0.98):
            seq_before = seq.copy()
            seq = _refine_cluster_2opt(seq, max_swaps=intra_refine_max_swaps)
            mean_e2, p90_e2, max_e2, med_e2 = _edge_stats(seq)
            did_refine = True
        else:
            mean_e2, p90_e2, max_e2, med_e2 = mean_e, p90_e, max_e, med_e

        if did_refine:
            LOGGER.debug(f"      [cluster {ci:03d}] refine: "
                         f"mean {mean_e:.4f}->{mean_e2:.4f}, p90 {p90_e:.4f}->{p90_e2:.4f}, "
                         f"max {max_e:.4f}->{max_e2:.4f}")
        return ci, seq, centroids[ci], (mean_e2, p90_e2, max_e2, med_e2)

    t0 = time.perf_counter()
    cluster_results = [None] * n_clusters
    if parallel and n_clusters > 1:
        max_workers = min(40, n_clusters)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_process_cluster, i) for i in range(n_clusters)]
            for fut in as_completed(futs):
                ci, lp, cvec, estats = fut.result()
                cluster_results[ci] = (lp, cvec, estats)
    else:
        for i in range(n_clusters):
            ci, lp, cvec, estats = _process_cluster(i)
            cluster_results[ci] = (lp, cvec, estats)
    t1 = time.perf_counter()
    local_paths = [cluster_results[i][0] for i in range(n_clusters)]
    centroids   = np.vstack([cluster_results[i][1] for i in range(n_clusters)])
    edge_stats  = np.vstack([cluster_results[i][2] for i in range(n_clusters)])  # (mean, p90, max, med)
    LOGGER.info(f"    - local paths done in {t1 - t0:.2f}s")
    LOGGER.info(f"    - intra refine stats over clusters: "
                f"mean edge mean={edge_stats[:,0].mean():.4f}, p90={np.percentile(edge_stats[:,1],50):.4f}, "
                f"max p95={np.percentile(edge_stats[:,2],95):.4f}, max max={edge_stats[:,2].max():.4f}")

    # узел->кластер
    for ci, seq in enumerate(local_paths):
        node2cluster[seq] = ci

    # ------------------ порядок кластеров: MST + DFS + polish ------------------
    def _pairwise_l2(A: np.ndarray) -> np.ndarray:
        g = (A * A).sum(axis=1, keepdims=True)
        D2 = g + g.T - 2.0 * (A @ A.T)
        np.maximum(D2, 0, out=D2)
        return np.sqrt(D2, dtype=np.float32)

    t0 = time.perf_counter()
    CD = _pairwise_l2(centroids)
    K = n_clusters
    in_mst = np.zeros(K, dtype=bool)
    parent = np.full(K, -1, dtype=np.int64)
    key = np.full(K, np.inf, dtype=np.float32)
    start = int(np.argmax(sizes)); key[start] = 0.0
    for _ in range(K):
        u = int(np.argmin(np.where(in_mst, np.inf, key))); in_mst[u] = True
        for v in range(K):
            if not in_mst[v] and CD[u, v] < key[v]:
                key[v] = CD[u, v]; parent[v] = u
    adj = [[] for _ in range(K)]
    for v in range(K):
        if parent[v] >= 0:
            u = int(parent[v]); adj[u].append(v); adj[v].append(u)
    for u in range(K):
        adj[u].sort(key=lambda v: CD[u, v])
    order_clusters: List[int] = []
    seen = np.zeros(K, dtype=bool)
    stack = [start]
    while stack:
        u = stack.pop()
        if seen[u]: continue
        seen[u] = True; order_clusters.append(u)
        for v in reversed(adj[u]):
            if not seen[v]: stack.append(v)
    t1 = time.perf_counter()
    LOGGER.info(f"    - cluster order (MST+DFS) ready in {t1 - t0:.2f}s")

    def _polish_cluster_order(orderC: List[int], passes: int = 2) -> List[int]:
        C = orderC[:]
        for _ in range(passes):
            improved = False
            for i in range(1, len(C) - 2):
                a, b, c = C[i-1], C[i], C[i+1]
                before = np.linalg.norm(centroids[a] - centroids[b]) + np.linalg.norm(centroids[b] - centroids[c])
                after  = np.linalg.norm(centroids[a] - centroids[c]) + np.linalg.norm(centroids[c] - centroids[b])
                if after + 1e-9 < before:
                    C[i], C[i+1] = C[i+1], C[i]
                    improved = True
            if not improved: break
        return C

    def _swap_adjacent_if_better(orderC: List[int]) -> List[int]:
        C = orderC[:]
        improved = False
        for i in range(1, len(C)-1):
            a, b = C[i], C[i+1]
            cur = np.linalg.norm(centroids[C[i-1]] - centroids[a]) + np.linalg.norm(centroids[a] - centroids[b])
            alt = np.linalg.norm(centroids[C[i-1]] - centroids[b]) + np.linalg.norm(centroids[b] - centroids[a])
            if alt + 1e-9 < cur:
                C[i], C[i+1] = C[i+1], C[i]
                improved = True
        if improved:
            LOGGER.info("    - cluster adjacency swap: improved neighbor pairs")
        return C

    t0 = time.perf_counter()
    order_clusters = _polish_cluster_order(order_clusters, passes=2)
    order_clusters = _swap_adjacent_if_better(order_clusters)
    centroid_path_len = float(np.sum(_pairwise_l2(centroids[order_clusters])))
    t1 = time.perf_counter()
    LOGGER.info(f"    - cluster-order polish: {centroid_path_len:.4f} total centroid-path in {t1 - t0:.3f}s")

    # ------------------ склейка: list-based + adaptive rotation + lookahead ------------------
    def _best_rotated_seq_to_tail(tail_node: int, seq: np.ndarray) -> Tuple[np.ndarray, float, bool, int]:
        """Подбираем ориентацию+ротацию для минимального моста tail→seq."""
        L = int(seq.size)
        if L == 0:
            return seq, 0.0, False, 0
        R = min(max(8, bridge_rotate_r), max(1, min(48, L // 4)))
        cand_pos = list(range(0, R)) + list(range(max(0, L - R), L))
        cand_pos = sorted(set(cand_pos))
        # keep
        starts_keep = np.array([seq[i] for i in cand_pos], dtype=np.int64)
        d_keep = np.linalg.norm(X[starts_keep] - X[tail_node], axis=1)
        # reverse
        seq_rev = seq[::-1]
        starts_rev = np.array([seq_rev[i] for i in cand_pos], dtype=np.int64)
        d_rev = np.linalg.norm(X[starts_rev] - X[tail_node], axis=1)
        ik, ir = int(np.argmin(d_keep)), int(np.argmin(d_rev))
        use_rev = d_rev[ir] < d_keep[ik]
        cut = cand_pos[ir if use_rev else ik]
        seq_use = (seq_rev if use_rev else seq)
        if cut > 0:
            seq_use = np.concatenate([seq_use[cut:], seq_use[:cut]])
        best = float(min(d_keep[ik], d_rev[ir]))
        return seq_use, best, use_rev, cut

    t0 = time.perf_counter()
    remaining = list(order_clusters)
    path_list: List[int] = []
    boundaries: List[int] = []  # индексы в глобальном порядке, где начинается новый кластер (для boundary-refine)
    reversed_blocks = 0
    bridges = []

    # старт
    if remaining:
        first = remaining.pop(0)
        seq0 = local_paths[first]
        path_list = list(seq0)
        boundaries.append(0)

    # greedy lookahead
    while remaining:
        tail_node = path_list[-1]
        w = min(merge_lookahead, len(remaining))
        best_j, best_d, best_seq, best_rev = 0, float("inf"), None, False
        for j in range(w):
            c = remaining[j]
            seq = local_paths[c]
            cand_seq, d, rev, cut = _best_rotated_seq_to_tail(tail_node, seq)
            if d < best_d:
                best_d, best_j, best_seq, best_rev = d, j, cand_seq, rev
        bridges.append(best_d)
        if best_rev:
            reversed_blocks += 1
        boundaries.append(len(path_list))  # начало нового блока
        path_list.extend(best_seq)
        remaining.pop(best_j)

    order = np.asarray(path_list, dtype=np.int64)
    t1 = time.perf_counter()
    if bridges:
        LOGGER.info(
            "    - merge (lookahead=%d) in %.2fs; reversed %d/%d (%.1f%%); "
            "bridge len avg=%.4f, p50=%.4f, p90=%.4f, max=%.4f"
            % (
                merge_lookahead, (t1 - t0), reversed_blocks, n_clusters,
                100.0 * reversed_blocks / max(1, n_clusters),
                float(np.mean(bridges)), float(np.percentile(bridges, 50)),
                float(np.percentile(bridges, 90)), float(np.max(bridges)),
            )
        )
    else:
        LOGGER.info(f"    - merge (lookahead={merge_lookahead}) in {t1 - t0:.2f}s")

    # ------------------ boundary-refine: микро 2-opt на границах ------------------
    def _boundary_refine(order: np.ndarray, Lw: int = 12, step_a: int = 4, step_b: int = 4) -> np.ndarray:
        if order.size < 4 or Lw <= 0:
            return order
        P = order.copy()
        B = sorted(set(boundaries[1:]))  # пропускаем первую (0)
        total_improved = 0
        for b in B:
            l = max(0, b - Lw)
            r = min(P.size, b + Lw)
            if r - l < 4: 
                continue
            seg = P[l:r]
            best_gain = 0.0; best_pair = None
            for a in range(1, seg.size - 2, step_a):
                for c in range(a + 10, seg.size - 1, step_b):
                    u1, v1 = seg[a - 1], seg[a]
                    u2, v2 = seg[c], (seg[c + 1] if c + 1 < seg.size else None)
                    before = np.linalg.norm(X[u1] - X[v1]) + (np.linalg.norm(X[u2] - X[v2]) if v2 is not None else 0.0)
                    after  = np.linalg.norm(X[u1] - X[u2]) + (np.linalg.norm(X[v1] - X[v2]) if v2 is not None else 0.0)
                    gain = before - after
                    if gain > best_gain:
                        best_gain, best_pair = gain, (l + a, l + c)
            if best_pair:
                a, c = best_pair
                P[a:c + 1] = P[a:c + 1][::-1]
                total_improved += 1
        LOGGER.info(f"    - boundary refine: processed {len(B)} joints, improved {total_improved}")
        return P

    t0 = time.perf_counter()
    if boundary_refine_L > 0:
        order = _boundary_refine(order, Lw=boundary_refine_L)
    t1 = time.perf_counter()
    if boundary_refine_L > 0:
        LOGGER.info(f"    - boundary refine done in {t1 - t0:.2f}s (L={boundary_refine_L})")

    # ------------------ финальная полировка по окнам ------------------
    def polish_joints(order: np.ndarray, window: int = 128, passes: int = 3) -> np.ndarray:
        Np = len(order)
        if Np < 4: return order
        P = order.copy()
        for _ in range(passes):
            i = 0
            while i < Np - 1:
                l = max(0, i - window // 2); r = min(Np, i + window // 2)
                seg = P[l:r]
                if seg.size < 4:
                    i += window // 2; continue
                best_gain, best_pair = 0.0, None
                for a in range(1, seg.size - 2, 6):
                    for b in range(a + 14, seg.size - 1, 6):
                        u1, v1 = seg[a - 1], seg[a]
                        u2 = seg[b]; v2 = seg[b + 1] if b + 1 < seg.size else None
                        before = np.linalg.norm(X[u1] - X[v1]) + (np.linalg.norm(X[u2] - X[v2]) if v2 is not None else 0.0)
                        after  = np.linalg.norm(X[u1] - X[u2]) + (np.linalg.norm(X[v1] - X[v2]) if v2 is not None else 0.0)
                        gain = before - after
                        if gain > best_gain:
                            best_gain, best_pair = gain, (l + a, l + b)
                if best_pair:
                    a, b = best_pair
                    P[a:b + 1] = P[a:b + 1][::-1]
                i += window // 2
        return P

    t0 = time.perf_counter()
    order = polish_joints(order, window=polish_window, passes=polish_passes)
    t1 = time.perf_counter()
    LOGGER.info(f"    - polish done in {t1 - t0:.2f}s (window={polish_window}, passes={polish_passes})")

    # ------------------ integrity-check и телеметрия ------------------
    uniq = np.unique(order)
    if uniq.size != N:
        counts = np.bincount(order, minlength=N)
        dups = np.flatnonzero(counts > 1)[:10]
        miss = np.setdiff1d(np.arange(N, dtype=np.int64), uniq)[:10]
        LOGGER.error(f"    - INTEGRITY FAIL: uniq={uniq.size}/{N}, dups(sample)={dups.tolist()}, missing(sample)={miss.tolist()}")
        seen = np.zeros(N, dtype=bool)
        clean = []
        for v in order:
            if not seen[v]:
                clean.append(v); seen[v] = True
        missing = np.where(~seen)[0]
        order = np.asarray(clean + missing.tolist(), dtype=np.int64)
        LOGGER.info(f"    - order repaired: now unique={order.size}")

    def _edge_lengths(order: np.ndarray) -> np.ndarray:
        if order.size < 2: return np.zeros(0, dtype=np.float32)
        P = X[order]
        return np.linalg.norm(P[1:] - P[:-1], axis=1).astype(np.float32)

    # def _top_jumps(order: np.ndarray, top: int = 10):
    #     el = _edge_lengths(order)
    #     if el.size == 0: return []
    #     idx = np.argsort(el)[::-1][:top]
    #     return [(int(i), float(el[i])) for i in idx]

    # def _top_intercluster(order: np.ndarray, top: int = 10):
    #     el = _edge_lengths(order)
    #     if el.size == 0: return []
    #     idx = np.argsort(el)[::-1][:top]
    #     out = []
    #     for i in idx:
    #         u, v = int(order[i]), int(order[i+1])
    #         out.append((int(i), float(el[i]), int(node2cluster[u]), int(node2cluster[v])))
    #     return out

    el = _edge_lengths(order)
    if el.size > 0:
        LOGGER.info(f"    - edge length stats: mean={el.mean():.4f}, p50={np.percentile(el,50):.4f}, "
                    f"p90={np.percentile(el,90):.4f}, p95={np.percentile(el,95):.4f}, max={el.max():.4f}")
        # LOGGER.info(f"    - top jumps (edge_idx, len): {_top_jumps(order, 10)}")
        # LOGGER.info(f"    - top inter-cluster jumps (edge_idx, len, c(u), c(v)): {_top_intercluster(order, 10)}")

    LOGGER.info(f"    - total time: {time.perf_counter() - t_all0:.2f}s")
    return order


import heapq

def _mst_adj_arrays_safe(mst: csr_matrix):
    """Смежность MST без дублей: берём только (u<v), добавляем двунаправленно, сортируем по весу."""
    M = mst.maximum(mst.T).tocoo()
    n = M.shape[0]
    adj = [[] for _ in range(n)]
    mask = M.row < M.col
    for u, v, w in zip(M.row[mask], M.col[mask], M.data[mask]):
        u = int(u); v = int(v); w = float(w)
        if u == v: 
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))
    for u in range(n):
        adj[u].sort(key=lambda x: x[1])
    return adj

def _graph_adj_arrays_symmetric(G: csr_matrix, max_neighbors: int = 16):
    """
    Смежность для общего графа (kNN): симметризуем, отсортировано по весу (возрастающе),
    обрезаем до max_neighbors на узел (чтобы не раздувать кандидаты).
    """
    GG = G.maximum(G.T).tocsr()
    n = GG.shape[0]
    adj = [[] for _ in range(n)]
    indptr, indices, data = GG.indptr, GG.indices, GG.data
    for u in range(n):
        start, end = indptr[u], indptr[u+1]
        neigh = [(int(indices[i]), float(data[i])) for i in range(start, end) if int(indices[i]) != u]
        neigh.sort(key=lambda x: x[1])
        if max_neighbors is not None and max_neighbors > 0:
            neigh = neigh[:max_neighbors]
        adj[u] = neigh
    return adj

def _tree_center_weighted(adj):
    """Центр (по диаметру) взвешенного дерева, два прохода."""
    def farthest(src):
        n = len(adj)
        dist = np.full(n, np.inf, dtype=np.float64)
        parent = np.full(n, -1, dtype=np.int64)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d,u = heapq.heappop(pq)
            if d > dist[u]: 
                continue
            for v,w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
        a = int(np.argmax(dist))
        return a, dist, parent

    a, _, _ = farthest(0)
    b, dist, parent = farthest(a)
    path = []
    cur = b
    while cur != -1:
        path.append(cur); cur = parent[cur]
    path.reverse()
    total = float(dist[b]); half = total/2.0
    acc = 0.0; center = path[0]
    for i in range(1, len(path)):
        u, v = path[i-1], path[i]
        w = next(w for to,w in adj[u] if to == v)
        if acc + w >= half:
            center = v if (half - acc) > (acc + w - half) else u
            break
        acc += w
    return int(center)

def _subtree_sizes(adj, root):
    """Размеры поддеревьев (rooted)."""
    n = len(adj)
    parent = np.full(n, -1, dtype=np.int64)
    order = []
    st = [root]; parent[root] = root
    while st:
        u = st.pop(); order.append(u)
        for v,_ in adj[u]:
            if parent[v] == -1:
                parent[v] = u; st.append(v)
    sizes = np.ones(n, dtype=np.int64)
    for u in reversed(order):
        for v,_ in adj[u]:
            if parent[v] == u:
                sizes[u] += sizes[v]
    parent[root] = -1
    return parent, sizes

def optimized_depth_first_order_v2(mst: csr_matrix,
                                   start_node: int = None,
                                   lookahead_depth: int = 2,
                                   beam_width: int = 3,
                                   progress_callback=None):
    """
    Улучшенный DFS по MST (без потери покрытия):
      • старт из центра дерева (если start_node=None),
      • соседи отсортированы по весу,
      • приоритет «тяжёлых» поддеревьев + 1-шаговый lookahead,
      • beam только задаёт ПОРЯДОК, но ВСЕ соседи посещаются (никакого прюнинга).
    Возвращает: (order, visited_mask) — order покрывает все вершины компоненты.
    """
    # 1) смежность
    adj = _mst_adj_arrays_safe(mst)
    n = len(adj)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.zeros(0, dtype=bool)

    # 2) старт
    if start_node is None or not (0 <= int(start_node) < n):
        start_node = _tree_center_weighted(adj)

    # 3) precompute sizes
    parent, sizes = _subtree_sizes(adj, start_node)

    visited = np.zeros(n, dtype=bool)
    order   = []

    def score(cur, nxt):
        # вес ребра
        w = next(w for v,w in adj[cur] if v == nxt)
        # «тяжесть» (больше поддерево — раньше идём)
        heavy = -float(sizes[nxt])
        # lookahead: минимальный вес ребра дальше
        mnext = np.inf
        for v,w2 in adj[nxt]:
            if not visited[v] and v != cur:
                mnext = w2
                break  # т.к. adj[nxt] отсортированы по весу
        if mnext is np.inf:
            mnext = 0.0
        return w + 0.3*mnext + 1e-4*heavy

    stack = [int(start_node)]
    processed_last = -1

    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)

        if progress_callback:
            processed = len(order)
            if processed != processed_last:
                progress_callback(processed)
                processed_last = processed

        # непосещённые соседи
        cands = [v for v,_ in adj[u] if not visited[v]]
        if not cands:
            continue

        # сортировка по score
        cands.sort(key=lambda v: score(u, v))
        # BEAM как приоритет: сначала top-beam, потом остальные
        bw = max(1, int(beam_width))
        primary = cands[:bw]
        backlog = cands[bw:]
        # кладём в стек так, чтобы первым пошёл лучший
        for v in reversed(backlog):
            stack.append(v)
        for v in reversed(primary):
            stack.append(v)

    if progress_callback:
        progress_callback(len(order), final_update=True)

    # safety: если вдруг что-то не дошли (не должно случиться на дереве) — допройдём plain-DFS
    if not np.all(visited):
        rem = np.where(~visited)[0].tolist()
        for s in rem:
            if visited[s]:
                continue
            # обычный стек без beam, только чтобы добрать компоненты
            st = [s]
            while st:
                u = st.pop()
                if visited[u]:
                    continue
                visited[u] = True
                order.append(u)
                for v,_ in adj[u]:
                    if not visited[v]:
                        st.append(v)

    return np.asarray(order, dtype=np.int64), visited

def dfs_polish(order_local, X, window=128, passes=2, focus_top=300):
    """
    Локальный 2-opt по окнам.
    Если focus_top>0 — сначала обрабатываем окна вокруг top-длинных рёбер.
    """
    N = order_local.size
    if N < 4:
        return order_local
    P = order_local.copy()

    # найдём топ длинных рёбер
    seg = np.linalg.norm(X[P][1:] - X[P][:-1], axis=1)
    if focus_top and seg.size > 0:
        idx = np.argsort(seg)[::-1][:min(focus_top, seg.size)]
        # превратим в список окон (l..r), мерджим пересекающиеся
        windows = []
        for i in idx:
            l = max(0, i - window//2)
            r = min(N, i + window//2)
            windows.append((l,r))
        # слияние интервалов
        windows.sort()
        merged = []
        for l,r in windows:
            if not merged or l > merged[-1][1]:
                merged.append([l,r])
            else:
                merged[-1][1] = max(merged[-1][1], r)
        win_list = [(l,r) for l,r in merged]
    else:
        win_list = [(0,N)]

    for _ in range(passes):
        for l,r in win_list:
            if r-l < 4: 
                continue
            seg_idx = np.arange(l, r)
            block = P[seg_idx]
            best_gain = 0.0; best = None
            # разрежённый перебор
            for a in range(1, block.size-2, 6):
                u1, v1 = block[a-1], block[a]
                for b in range(a+14, block.size-1, 6):
                    u2, v2 = block[b], block[b+1] if b+1 < block.size else None
                    before = np.linalg.norm(X[u1]-X[v1]) + (np.linalg.norm(X[u2]-X[v2]) if v2 is not None else 0.0)
                    after  = np.linalg.norm(X[u1]-X[u2]) + (np.linalg.norm(X[v1]-X[v2]) if v2 is not None else 0.0)
                    gain = before - after
                    if gain > best_gain:
                        best_gain = gain; best = (l+a, l+b)
            if best:
                a, b = best
                P[a:b+1] = P[a:b+1][::-1]
    return P



def sort_by_ann_mst(feats: np.ndarray, k: int, config: Config):
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
        LOGGER.error("! ОШИБКА: Библиотека 'faiss' не установлена. Сортировка невозможна.")
        return None

    n, d = feats.shape
    total_start_time = time.time()


    LOGGER.info("\n" + "="*80)
    LOGGER.info(f"Запуск улучшенной сортировки методом ANN + MST (на основе Евклидова расстояния)") # Изменено для ясности
    LOGGER.info(f"  - Изображений: {n}")
    LOGGER.info(f"  - Размерность фичей: {d}")
    LOGGER.info(f"  - Соседей на точку (k): {k}")
    LOGGER.info(f"  - Размер батча: {config.search.global_knn_batch_size}")
    LOGGER.info(f"  - Использовать GPU: {not config.model.use_cpu}")
    LOGGER.info(f"  - Оптимизатор: {config.sorting.optimizer} (block_size={config.sorting.two_opt_block_size}, shift={config.sorting.two_opt_shift})")
    LOGGER.info("="*80)

    # Работаем с копией float32 для FAISS
    feats_copy = feats.astype('float32').copy()

    # --- Шаг 1: Индексация в FAISS ---
    step_start_time = time.time()
    LOGGER.info(f"\n[1/5] Шаг 1: Индексация векторов в FAISS...")
    try:
        # ИЗМЕНЕНИЕ 1: Убираем нормализацию. Она не нужна для евклидова расстояния.
        # faiss.normalize_L2(feats_copy)

        # ИЗМЕНЕНИЕ 2: Используем IndexFlatL2 для евклидова расстояния вместо IndexFlatIP.
        index = faiss.IndexFlatL2(d)

        if not config.model.use_cpu and torch is not None and torch.cuda.is_available():
            LOGGER.info("  - Попытка использовать GPU для FAISS...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            LOGGER.info("  - Индекс успешно перенесен на GPU.")

        index.add(feats_copy)
        LOGGER.info(f"  - Индекс создан. Всего векторов: {index.ntotal}.")
        LOGGER.info(f"  - Время на шаг 1: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        LOGGER.error(f"! Ошибка на шаге 1: {e}")
        return None

    # --- Шаг 2: Поиск k-ближайших соседей ---
    step_start_time = time.time()
    LOGGER.info(f"\n[2/5] Шаг 2: Поиск {k} ближайших соседей...")
    try:
        raw_batch_size = getattr(config.search, "global_knn_batch_size", 1)
        try:
            batch_size = int(raw_batch_size)
        except (TypeError, ValueError):
            batch_size = 1
        batch_size = max(1, batch_size)
        batch_ranges = [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]

        distances_sq = np.empty((n, k + 1), dtype=np.float32)
        indices = np.empty((n, k + 1), dtype=np.int64)

        def search_batch(start: int, end: int):
            """Execute FAISS search for the given slice in a worker thread."""
            distances_batch, indices_batch = index.search(feats_copy[start:end], k + 1)
            return start, end, distances_batch, indices_batch

        requested_workers = getattr(config.sorting, "knn_workers", None)
        try:
            requested_workers = int(requested_workers)
        except (TypeError, ValueError):
            requested_workers = 0

        if requested_workers <= 0:
            cpu_count = os.cpu_count() or 1
            max_workers = min(len(batch_ranges), cpu_count)
        else:
            max_workers = min(len(batch_ranges), max(1, requested_workers))
        max_workers = max(1, max_workers)

        LOGGER.info(f"  - K-NN worker threads: {max_workers}")

        if not batch_ranges:
            distances_sq = np.empty((0, k + 1), dtype=np.float32)
            indices = np.empty((0, k + 1), dtype=np.int64)
        elif max_workers == 1:
            for start, end in tqdm(batch_ranges, desc="  - Поиск k-NN (батчи)"):
                start_idx, end_idx, distances_batch, indices_batch = search_batch(start, end)
                distances_sq[start_idx:end_idx] = np.asarray(distances_batch, dtype=np.float32)
                indices[start_idx:end_idx] = np.asarray(indices_batch, dtype=np.int64)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(search_batch, start, end) for start, end in batch_ranges]
                with tqdm(total=len(futures), desc="  - Поиск k-NN (батчи)") as progress_bar:
                    for future in as_completed(futures):
                        start, end, distances_batch, indices_batch = future.result()
                        distances_sq[start:end] = np.asarray(distances_batch, dtype=np.float32)
                        indices[start:end] = np.asarray(indices_batch, dtype=np.int64)
                        progress_bar.update(1)

        LOGGER.info(f"  - Поиск завершен. Размер матрицы индексов: {indices.shape}")
        LOGGER.info(f"  - Время на шаг 2: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        LOGGER.error(f"! Ошибка на шаге 2: {e}")
        return None

    # --- Шаг 3: Построение симметричного разреженного графа ---
    step_start_time = time.time()
    LOGGER.info(f"\n[3/5] Шаг 3: Построение симметричного графа...")
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

        LOGGER.info(f"  - Граф успешно создан. Количество ребер: {symmetric_graph.nnz}.")
        LOGGER.info(f"  - Время на шаг 3: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        LOGGER.error(f"! Ошибка на шаге 3: {e}")
        return None

    # --- Шаги 4 и 5 остаются без изменений, так как они работают с графом,
    # которому неважно, как были получены веса ребер. ---

    # --- Шаг 4: Построение Minimum Spanning Tree (MST) ---
    step_start_time = time.time()
    LOGGER.info(f"\n[4/5] Шаг 4: Построение Minimum Spanning Tree...")
    try:
        mst = minimum_spanning_tree(symmetric_graph)
        LOGGER.info(f"  - MST построено. Общая стоимость дерева: {mst.sum():.2f}.")
        LOGGER.info(f"  - Время на шаг 4: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        LOGGER.error(f"! Ошибка на шаге 4: {e}")
        return None

    # --- Шаг 5


    # --- Начало исправленного блока для Шага 5 ---
    step_start_time = time.time()
    LOGGER.info(f"\n[5/5] Шаг 5: Двухэтапная кластеризация и обход...")
    
    try:
        n = mst.shape[0]
        # Определяем все связанные компоненты в исходном MST
        n_components, labels = connected_components(csgraph=mst, directed=False, return_labels=True)
        
        # --- ЭТАП 1: ОБРАБОТКА ОСНОВНОЙ КОМПОНЕНТЫ ---
        LOGGER.info(f"\n  --- Этап 1: Обработка основной компоненты ---")
        if n_components > 0:
            component_sizes = np.bincount(labels)
            main_component_id = np.argmax(component_sizes)
            main_nodes_mask = (labels == main_component_id)
            main_nodes_indices = np.where(main_nodes_mask)[0]
            total_in_main_component = len(main_nodes_indices)
            
            LOGGER.info(f"  - Найдена основная компонента размером {total_in_main_component} узлов.")
            
            start_node = main_nodes_indices[0]
            
            # --- выбор стратегии для основной компоненты ---
            strategy = str(getattr(config.sorting, "strategy", "dfs")).lower()

            # сабматрица MST по основной компоненте (локальные индексы 0..M-1)
            mst_main = mst[main_nodes_indices, :][:, main_nodes_indices]
            X_main   = feats_copy[main_nodes_indices].astype(np.float32, copy=False)
            M        = mst_main.shape[0]

            # безопасный прогресс-логгер (без nonlocal/global)
            def make_progress_logger(total, logger):
                state = {"last": -1}
                def report(processed_count, final_update=False):
                    pct = int((processed_count * 100) / max(1, total))
                    if pct > state["last"] or final_update:
                        logger.info(f"    - Прогресс: {processed_count} / {total} узлов ({pct}%)", end='\r')
                        state["last"] = pct
                return report

            report_progress_local = make_progress_logger(M, LOGGER)

            order_local = None

            if strategy == "dfs":
                LOOKAHEAD_DEPTH = int(getattr(config.sorting, "lookahead", 1))
                if LOOKAHEAD_DEPTH <= 1:
                    from scipy.sparse.csgraph import depth_first_order
                    LOGGER.info("  - Стратегия 'dfs' с глубиной <= 1. Стандартный DFS (локально по компоненте).")
                    # старт в локальных координатах (если глобальный задан — приведём)
                    try:
                        if start_node is not None:
                            pos = np.where(main_nodes_indices == int(start_node))[0]
                            start_local = int(pos[0]) if pos.size > 0 else 0
                        else:
                            start_local = 0
                    except Exception:
                        start_local = 0
                    main_path_indices, _ = depth_first_order(mst_main, i_start=start_local, directed=False)
                    order_local = main_path_indices[main_path_indices != -1]
                    LOGGER.info("    - Стандартный DFS завершен.")
                else:
                    LOGGER.info(f"  - Стратегия 'dfs' с глубиной {LOOKAHEAD_DEPTH}. DFS v2 (центр+heavy-path+beam).")
                    beam_width = int(getattr(config.sorting, "beam_width", 3))
                    order_local, visited = optimized_depth_first_order_v2(
                        mst_main,
                        start_node=None,
                        lookahead_depth=LOOKAHEAD_DEPTH,
                        beam_width=beam_width,
                        progress_callback=report_progress_local
                    )
                    LOGGER.info("")  # перенос строки
                    if not np.all(visited):
                        missing = np.flatnonzero(~visited)
                        LOGGER.warning(f"    - dfs v2: добираем {missing.size} узлов.")
                        order_local = np.concatenate([order_local, missing])
            elif strategy == "farthest_insertion":

                LOGGER.info("  - Стратегия 'farthest_insertion'. Запуск...")
                order_local = farthest_insertion_path_clustered(X_main, config)
                LOGGER.info("    - Farthest Insertion завершен.")

            else:
                LOGGER.info(f"  - Стратегия '{strategy}' не распознана. Фоллбек: стандартный DFS.")
                from scipy.sparse.csgraph import depth_first_order
                main_path_indices, _ = depth_first_order(mst_main, i_start=0, directed=False)
                order_local = main_path_indices[main_path_indices != -1]

            # --- опциональная полировка пути для dfs-веток ---
            enable_polish = bool(getattr(config.sorting, "enable_dfs_polish", False))
            if enable_polish and strategy == "dfs":
                pw   = int(getattr(config.sorting, "dfs_polish_window", 136))
                pp   = int(getattr(config.sorting, "dfs_polish_passes", 2))
                ftop = int(getattr(config.sorting, "dfs_polish_focus_top", 300))
                if order_local.size > 1:
                    pre_len = float(np.linalg.norm(X_main[order_local][1:] - X_main[order_local][:-1], axis=1).sum())
                order_local = dfs_polish(order_local, X_main, window=pw, passes=pp, focus_top=ftop)
                if order_local.size > 1:
                    post_len = float(np.linalg.norm(X_main[order_local][1:] - X_main[order_local][:-1], axis=1).sum())
                    LOGGER.info(f"    - dfs-polish: window={pw}, passes={pp}, top={ftop}; Δsum={pre_len - post_len:.3f}")

            # --- интегрити-чек локального порядка ---
            if order_local is None or order_local.size != M:
                LOGGER.error(f"    - INTEGRITY: размер порядка {0 if order_local is None else order_local.size} != {M}; чиню.")
                if order_local is None:
                    order_local = np.arange(M, dtype=np.int64)
                else:
                    seen = np.zeros(M, dtype=bool)
                    clean = []
                    for v in order_local:
                        if 0 <= v < M and not seen[v]:
                            clean.append(int(v)); seen[v] = True
                    missing = np.where(~seen)[0]
                    order_local = np.asarray(clean + missing.tolist(), dtype=np.int64)

            if order_local.min() < 0 or order_local.max() >= M:
                LOGGER.error(f"    - INTEGRITY: значения вне [0,{M-1}] (min={order_local.min()}, max={order_local.max()}); clip.")
                order_local = np.clip(order_local, 0, M-1)

            # --- метрики считаем ЛОКАЛЬНО ---
            metrics = evaluate_order_metrics(order=order_local, X=X_main, labels=None)
            print(metrics)

            # --- глобальный путь для последующих шагов ---
            main_path = main_nodes_indices[order_local]
            LOGGER.info(f"  - Основная компонента обработана. Длина основного пути: {len(main_path)}.")



        else:
            # Редкий случай, если граф пуст
            LOGGER.info("  - Не найдено ни одной компоненты. Основной путь пуст.")
            main_path = np.array([], dtype=int)
            main_nodes_mask = np.zeros(n, dtype=bool)
    
        # --- ЭТАП 2: ОБРАБОТКА ОСТАВШИХСЯ "ОСТРОВОВ" ---
        LOGGER.info(f"\n  --- Этап 2: Обработка оставшихся компонент ('островов') ---")
        
        island_nodes_indices = np.where(~main_nodes_mask)[0]
        num_islands = len(island_nodes_indices)
        
        secondary_path = np.array([], dtype=int) # Инициализируем вторичный путь
    
        if num_islands > 1:
            LOGGER.info(f"  - Найдено {num_islands} узлов в 'островах'. Запускаем для них отдельный процесс ANN+MST+DFS.")
            
            # Шаг 2.1: Извлекаем фичи только для "островов"
            island_feats = feats_copy[island_nodes_indices]
            
            # Шаг 2.2: ANN. Строим граф "все ко всем" (k = N-1)
            LOGGER.info(f"    - [2.1/2.4] Поиск соседей для {num_islands}...")
            island_k = min(1000, num_islands - 1) # k не может быть больше N-1
            island_index = faiss.IndexFlatL2(island_feats.shape[1])
            # if config.model.use_cpu is None: # Предполагается, что use_gpu определена ранее
            #     res = faiss.StandardGpuResources()
            #     island_index = faiss.index_cpu_to_gpu(res, 0, island_index)
            island_index.add(island_feats)
            island_dists, island_neighbors_local_idx = island_index.search(island_feats, k=island_k)
            
            # Шаг 2.3: Построение графа и MST для "островов"
            LOGGER.info(f"    - [2.2/2.4] Построение графа для 'островов'...")
            
            # Индексы из FAISS - локальные (от 0 до num_islands-1).
            rows = np.arange(num_islands).repeat(island_k)
            cols = island_neighbors_local_idx.flatten()
            data = island_dists.flatten()
            
            # Убираем петли (i,i) и некорректные расстояния
            valid_mask = (rows != cols) & (data > 0)
            island_graph = coo_matrix((data[valid_mask], (rows[valid_mask], cols[valid_mask])), shape=(num_islands, num_islands))
            island_graph.eliminate_zeros()
            
            LOGGER.info(f"    - [2.3/2.4] Построение MST для 'островов'...")
            island_mst = minimum_spanning_tree(island_graph)
            
            # Шаг 2.4: Простой DFS для MST "островов"
            LOGGER.info(f"    - [2.4/2.4] Запуск простого DFS для 'островов'...")
            from scipy.sparse.csgraph import depth_first_order
            
            # Начинаем с первого узла в локальном списке "островов"
            island_path_local, _ = depth_first_order(island_mst, i_start=0, directed=False)
            island_path_local_visited = island_path_local[island_path_local != -1]
            
            # Преобразуем локальные индексы пути (0..num_islands-1) в глобальные индексы изображений
            secondary_path = island_nodes_indices[island_path_local_visited]
            LOGGER.info(f"  - 'Острова' обработаны. Длина вторичного пути: {len(secondary_path)}.")
            
        elif num_islands == 1:
            LOGGER.info("  - Найден 1 узел в 'островах', добавляем его напрямую.")
            secondary_path = island_nodes_indices
        else:
            LOGGER.info("  - Нет 'островов' для обработки.")
    
        # --- ЭТАП 3: ФИНАЛЬНАЯ СБОРКА И ПРОВЕРКА ---
        LOGGER.info(f"\n  --- Этап 3: Финальная сборка пути ---")
        
        # Собираем основной путь и путь "островов"
        path_list = list(main_path) + list(secondary_path)
        
        # Проверяем, все ли узлы были посещены. Добавляем "потерянные" в конец.
        if len(path_list) != n:
            LOGGER.error(f"  - ! ПРЕДУПРЕЖДЕНИЕ: Длина пути ({len(path_list)}) не равна общему числу элементов ({n}).")
            all_nodes = set(range(n))
            path_nodes = set(path_list)
            unvisited = list(all_nodes - path_nodes)
            LOGGER.info(f"    - Найдено {len(unvisited)} непосещенных узлов. Добавляем их в конец.")
            path_list.extend(unvisited)
        
        path = np.array(path_list)
        
        LOGGER.info(f"  - Сборка завершена. Получен полный путь из {len(path)} элементов.")
        LOGGER.info(f"  - Время на шаг 5: {time.time() - step_start_time:.2f} сек.")
    
    except Exception as e:
        LOGGER.error(f"\n! КРИТИЧЕСКАЯ ОШИБКА на шаге 5: {e}")
        import traceback
        traceback.print_exc()
        path = None # Возвращаем None в случае ошибки
    
   
    # --- Шаг 6: Пост-обработка пути с оптимизатором ---
    step_start_time = time.time()
    LOGGER.info(f"\n[6/6] Шаг 6: Пост-обработка пути с {config.sorting.optimizer} (блоки {config.sorting.two_opt_block_size}, сдвиг {config.sorting.two_opt_shift})...")
    
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
    num_blocks = max(1, (n - config.sorting.two_opt_block_size) // config.sorting.two_opt_shift + 1)
    
    for b in tqdm(range(num_blocks), desc="  - Оптимизация блоков"):
        start = b * config.sorting.two_opt_shift
        end = min(start + config.sorting.two_opt_block_size, n)
        if end - start < 3:
            continue
        
        subpath = optimized_path[start:end]
        sub_feats = feats[subpath]
        
        # В вашей логике концы всегда фиксированы, кроме, возможно, крайних блоков.
        # Для простоты здесь всегда считаем их фиксированными, т.к. они соединяются с остальной частью пути.
        # Если нужна особая логика для крайних блоков, ее можно добавить сюда.
        
        if config.sorting.optimizer == '2opt':
            optimized_path[start:end] = two_opt(subpath, sub_feats)

    
    path = optimized_path
    LOGGER.info(f"  - Пост-обработка завершена. Время: {time.time() - step_start_time:.2f} сек.")

    
    # --- Финальный вывод (замените ваши оригинальные print'ы на это, чтобы учесть шаг 6) ---
    
    LOGGER.info("\n" + "="*80)
    LOGGER.info("Сортировка методом ANN + MST успешно завершена.")
    LOGGER.info(f"Общее время выполнения: {time.time() - total_start_time:.2f} сек.")
    LOGGER.info("="*80 + "\n")
    
    return path

def sort_images(feats, paths, config: Config):
    n = feats.shape[0]
    if n < 1:
        LOGGER.info("Нет изображений для сортировки.")
        return

    
    LOGGER.info(f"\n# {'-'*76} #")
    LOGGER.info(f"# Сортировка изображений методом 'ANN+MST'") 
    LOGGER.info(f"# {'-'*76} #")
    LOGGER.info(f"Всего изображений: {n}")

    # Подбираем разумное k для внутреннего графа (как раньше)
    if len(paths) < config.sorting.neighbors_k_limit:
        k_neighbors = len(paths) - 1
    else:
        k_neighbors = config.sorting.neighbors_k_limit

    use_gpu_faiss = not config.model.use_cpu and torch.cuda.is_available()
    final_order = sort_by_ann_mst(feats, k_neighbors, config)

    if final_order is None or len(final_order) == 0:
        LOGGER.error("! Сортировка не удалась, итоговый путь пуст. Операция отменена.")
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
    save_faiss_index(index,config.files.index_file,config)
    
    # сохраняем mapping (faiss position -> original index)
    np.save(config.files.index_file + ".order.npy", ordered_indices)   # файл: faiss.index.order.npy
    
    # (опционально) сохраним также список путей в том же порядке — удобно для отладки
    ordered_paths = [paths[i] for i in ordered_indices]
    with open(config.files.index_file + ".paths.txt", "w", encoding="utf-8") as f:
        for p in ordered_paths:
            f.write(p + "\n")


    # Если пользователь запросил только список — формируем TSV с глобальными соседями
    if config.misc.list_only:
        # Вычисляем глобальные k-NN (по всей базе). Используем args.neighbors
        knn_k = max(1, config.search.tsv_neighbors)  # минимум 1 сосед
        LOGGER.info(f"\nВычисляем глобальные {knn_k} ближайших соседей и формируем файл...")

        # Сначала постройте таблицу обратной индексации: original_index -> position in final order
        inverse_order = np.empty(len(final_order), dtype=int)
        for pos, orig_idx in enumerate(final_order):
            inverse_order[orig_idx] = pos

        # Выполняем глобальный поиск соседей (возвращаются индексы по оригинальным индексам)
        knn_idxs, knn_dists = compute_global_knn(feats, knn_k, config, use_gpu=use_gpu_faiss)

        out_file = config.files.out_tsv
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
    copy_and_rename(paths, final_order, config.files.dst_folder, config)
    LOGGER.info("Сортировка завершена.")
