import time
import math
import numpy as np
from typing import Dict, Optional, Callable, Tuple

# ---------- базовые утилиты ----------
def path_length(order: np.ndarray, X: np.ndarray) -> float:
    P = X[order]
    dif = P[1:] - P[:-1]
    seg = np.sqrt(np.sum(dif * dif, axis=1))
    return float(np.sum(seg))

def approx_pair_spearman(order: np.ndarray, X: np.ndarray, pairs: int = 50000, seed: int = 42) -> float:
    print("approx_pair_spearman")
    """
    Корреляция Спирмена между:
      - расстоянием вдоль пути (|pos[i]-pos[j]|)
      - L2 в пространстве
    Сэмплим пары, чтобы не было O(N^2).
    """
    rng = np.random.default_rng(seed)
    N = len(order)
    pos = np.empty(N, dtype=np.int32)
    pos[order] = np.arange(N, dtype=np.int32)

    I = rng.integers(0, N, size=(pairs, 2))
    I[:, 1] = (I[:, 1] + (I[:, 1] == I[:, 0])) % N  # избегаем одинаковых

    pi = I[:, 0]; pj = I[:, 1]
    d_path = np.abs(pos[pi] - pos[pj]).astype(np.float32)

    V = X[pi] - X[pj]
    d_l2 = np.sqrt(np.sum(V * V, axis=1)).astype(np.float32)

    # ранги
    def _ranks(a: np.ndarray) -> np.ndarray:
        idx = np.argsort(a)
        r = np.empty_like(idx, dtype=np.float32)
        r[idx] = np.arange(len(a), dtype=np.float32)
        return r

    r1 = _ranks(d_path)
    r2 = _ranks(d_l2)
    r1_center = r1 - r1.mean()
    r2_center = r2 - r2.mean()
    num = float(np.dot(r1_center, r2_center))
    den = float(np.linalg.norm(r1_center) * np.linalg.norm(r2_center) + 1e-12)
    return num / den

def knn_recall_at_window(order: np.ndarray,
                         X: np.ndarray,
                         k: int = 20,
                         window: int = 100,
                         sample: int = 5000,
                         seed: int = 42,
                         use_faiss: bool = True,
                         max_mem_gb: float = 1.0) -> float:
    print("knn_recall_at_window")
    """
    Доля k-ближайших соседей по L2, попавших в окно ±window вокруг позиции в порядке.
    Память-эффективная реализация (FAISS или матричная формула без 3D broadcast).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    N, D = X.shape
    sample = int(min(sample, N))

    chosen = rng.choice(N, size=sample, replace=False)
    pos = np.empty(N, dtype=np.int32)
    pos[order] = np.arange(N, dtype=np.int32)

    # ---------- быстрый путь: FAISS ----------
    if use_faiss:
        try:
            import faiss
            xb = X.astype(np.float32, copy=False)
            index = faiss.IndexFlatL2(D)
            index.add(xb)
            # Ищем k+1, чтобы отбросить саму точку при совпадении
            # Будем делать поиск батчами, чтобы не держать всё в ОЗУ
            bs = 4096
            all_neighbors = []
            for s in range(0, sample, bs):
                e = min(s + bs, sample)
                q = xb[chosen[s:e]]
                _, I = index.search(q, k + 1)
                # удаляем self
                cleaned = []
                for row, qi in zip(I, chosen[s:e]):
                    row = row[row != qi]
                    if row.shape[0] > k:
                        row = row[:k]
                    cleaned.append(row)
                all_neighbors.extend(cleaned)
            # Подсчёт recall
            hits = 0
            total = 0
            for qi, neigh in zip(chosen, all_neighbors):
                p = int(pos[qi])
                left = max(0, p - window)
                right = min(N - 1, p + window)
                win = set(order[left:right + 1])
                win.discard(qi)
                hits += sum(1 for t in neigh if t in win)
                total += len(neigh)
            return hits / max(1, total)
        except Exception:
            # упадём на fallback
            pass

    # ---------- fallback: матричная формула (чанками), без 3D broadcast ----------
    xb = X.astype(np.float32, copy=False)
    x2 = np.sum(xb * xb, axis=1)  # (N,)
    # выберем batch так, чтобы матрица (batch×N) поместилась в max_mem_gb
    # потребление ≈ batch * N * 4 байт
    bytes_limit = max_mem_gb * (1024 ** 3)
    batch = max(256, int(bytes_limit // (4 * max(1, N))))
    batch = min(batch, sample)

    hits = 0
    total = 0
    for s in range(0, sample, batch):
        e = min(s + batch, sample)
        Q = xb[chosen[s:e]]                       # (b,D)
        q2 = np.sum(Q * Q, axis=1, keepdims=True) # (b,1)
        # матрица расстояний без третьей оси: d2 = q2 + x2 - 2 Q X^T
        D2 = q2 + x2[None, :] - 2.0 * (Q @ xb.T)  # (b,N)
        # чтобы не выбирать self
        for irow, qi in enumerate(chosen[s:e]):
            D2[irow, qi] = np.inf
        # топ-(k) индексы (argpartition по строкам)
        idx = np.argpartition(D2, kth=k, axis=1)[:, :k]
        # подчистим точные топ-k в этих колонках
        row_off = np.arange(e - s)[:, None]
        D2_small = D2[row_off, idx]
        ord_small = np.argsort(D2_small, axis=1)
        topk = idx[row_off, ord_small][:, :k]     # (b,k)

        for (qi, neigh_row) in zip(chosen[s:e], topk):
            p = int(pos[qi])
            left = max(0, p - window)
            right = min(N - 1, p + window)
            win = set(order[left:right + 1])
            win.discard(qi)
            hits += sum(1 for t in neigh_row if t in win)
            total += neigh_row.shape[0]

    return hits / max(1, total)


def trustworthiness(order: np.ndarray, X: np.ndarray, k: int = 20, sample: int = 3000, seed: int = 42) -> float:
    print("trustworthiness")
    """
    Trustworthiness@k: 1 - (2/(n*k*(2n-3k-1))) * sum over i sum_{j in U_i} (rank_X(i,j) - k)
    где U_i — точки, которые попали в k-ближайших по ПОРЯДКУ (окрестность по порядку),
    но не входят в k-ближайших по пространству. Здесь используем «симметричное» окно по порядку.
    """
    rng = np.random.default_rng(seed)
    N = len(order)
    sample = min(sample, N)
    chosen = rng.choice(N, size=sample, replace=False)

    # ранги по L2 для выбранных относительно всего массива (только топ k*8, чтобы было быстрее)
    pos = np.empty(N, dtype=np.int32)
    pos[order] = np.arange(N, dtype=np.int32)

    # зададим «соседей по порядку» как окно ±k
    t_sum = 0.0
    max_rank = k * 8  # ограничим глубину ранжирования
    for qi in chosen:
        p = int(pos[qi])
        ord_neighbors = set(order[max(0, p - k):min(N, p + k + 1)])
        ord_neighbors.discard(qi)

        # топ M по L2 (M=max_rank), чтобы иметь ранги на уровне k..M
        dif = X - X[qi]
        d2 = np.sum(dif * dif, axis=1)
        # удаляем self
        d2[qi] = np.inf
        idx = np.argpartition(d2, kth=max_rank)[:max_rank]
        # полные ранги в пределах idx
        ranks_part = idx[np.argsort(d2[idx], kind='mergesort')]
        rank_map = {int(j): r+1 for r, j in enumerate(ranks_part)}  # ранги с 1

        # «ложные соседи» — те, кто в ord_neighbors, но не входят в топ-k по X
        # (или входят, но мы должны их исключить из штрафа)
        u_i = [j for j in ord_neighbors if rank_map.get(int(j), max_rank+1) > k]
        # штраф — sum(rank(i,j) - k)
        for j in u_i:
            r = rank_map.get(int(j), max_rank+1)
            t_sum += (r - k)

    n = float(sample)
    # нормировочный множитель
    norm = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))
    T = 1.0 - norm * t_sum
    return float(max(0.0, min(1.0, T)))

def label_run_length(order: np.ndarray, labels: np.ndarray) -> float:
    print("label_run_length")
    """Средняя длина пробега (run length) одинаковой метки по порядку."""
    if labels is None:
        return np.nan
    labels = np.asarray(labels)
    s = 1
    runs = []
    for i in range(1, len(order)):
        if labels[order[i]] == labels[order[i-1]]:
            s += 1
        else:
            runs.append(s)
            s = 1
    runs.append(s)
    return float(np.mean(runs))

def sequential_map_at_k(order: np.ndarray, labels: np.ndarray, K: int = 50, sample: int = 2000, seed: int = 42) -> float:
    print("sequential_map_at_k")
    """
    mAP, если поиск релевантных (по метке) делаем линейным сканированием окна ±K вокруг позиции запроса.
    """
    if labels is None:
        return np.nan
    rng = np.random.default_rng(seed)
    N = len(order)
    sample = min(sample, N)
    chosen = rng.choice(N, size=sample, replace=False)
    pos = np.empty(N, dtype=np.int32)
    pos[order] = np.arange(N, dtype=np.int32)

    aps = []
    for qi in chosen:
        y = labels[qi]
        p = int(pos[qi])
        cand = order[max(0, p - K):min(N, p + K + 1)]
        # исключаем сам запрос
        cand = cand[cand != qi]
        rel = (labels[cand] == y).astype(np.int32)
        if rel.sum() == 0:
            continue
        # AP по бинарной релевантности
        cum_rel = np.cumsum(rel)
        prec = cum_rel / (np.arange(len(rel)) + 1.0)
        ap = float(np.sum(prec[rel == 1]) / rel.sum())
        aps.append(ap)
    return float(np.mean(aps)) if aps else np.nan

# ---------- агрегатор ----------
def evaluate_order_metrics(order: np.ndarray, X: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
    return {
        "path_length": path_length(order, X),
        "spearman_pair_corr": approx_pair_spearman(order, X, pairs=50000),
        "knn_recall@20_w100": knn_recall_at_window(order, X, k=20, window=100, sample=5000),
        "label_runlen": label_run_length(order, labels) if labels is not None else np.nan,
        "sequential_mAP@50": sequential_map_at_k(order, labels, K=50, sample=2000) if labels is not None else np.nan,
        # "trustworthiness@20": trustworthiness(order, X, k=20, sample=3000),
    }

# ---------- бенч двух алгоритмов ----------
def benchmark_algorithms(
    X: np.ndarray,
    build_a: Callable[[], np.ndarray],
    build_b: Callable[[], np.ndarray],
    labels: Optional[np.ndarray] = None,
    names: Tuple[str, str] = ("A", "B"),
) -> Dict[str, Dict[str, float]]:
    t0 = time.perf_counter()
    order_a = build_a()
    t1 = time.perf_counter()
    order_b = build_b()
    t2 = time.perf_counter()

    m_a = evaluate_order_metrics(order_a, X, labels)
    m_b = evaluate_order_metrics(order_b, X, labels)

    m_a["build_seconds"] = t1 - t0
    m_b["build_seconds"] = t2 - t1

    return {names[0]: m_a, names[1]: m_b}
