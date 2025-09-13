import os
import shutil

import numpy as np
import tqdm
from cli import ARG_LOG_LEVEL


def copy_and_rename(paths, order, out_folder):
    """Копирует и переименовывает файлы в соответствии с вычисленным порядком."""
    # Создаем папку, если она не существует. Если существует - ничего не делаем.
    os.makedirs(out_folder, exist_ok=True)

    total_files = len(order)
    # -1, потому что индексация с нуля. Если файлов 100, то номера от 0 до 99.
    num_digits = len(str(total_files - 1)) if total_files > 0 else 1 
    fmt = f"{{:0{num_digits}d}}"

    if ARG_LOG_LEVEL == "default":
        pbar = tqdm(total=total_files, desc=f"Копирование в '{out_folder}'")
        for new_i, old_i in enumerate(order):
            src = paths[old_i]
            ext = os.path.splitext(src)[1].lower()
            original_name = os.path.splitext(os.path.basename(src))[0]
            # Формат имени: 0000_original_name.ext для удобной сортировки
            dst = os.path.join(out_folder, f"{fmt.format(new_i)}_{original_name}{ext}")
            shutil.copy2(src, dst)
            pbar.update(1)
        pbar.close()

    if ARG_LOG_LEVEL == "default": print(f"Копирование завершено.")

def output_sequence_with_neighbors(paths, feats, order, neighbors=3, out_file=None):
    """
    Выводит (или сохраняет) последовательность файлов в порядке `order` и
    для каждой позиции показывает до `neighbors` соседей слева и справа
    в формате index:distance,index:distance.

    - paths: list of original file paths (в том же порядке, в котором были загружены фичи)
    - feats: np.ndarray shape (N, D) — массив признаков (соответствует paths)
    - order: np.ndarray shape (N,) — порядок индексов (относительно feats/paths)
    - neighbors: int — сколько соседей слева и справа выводить
    - out_file: str|None — путь для сохранения (utf-8). Если None — вывод в stdout.
    """

    n = len(order)
    if n == 0:
        if ARG_LOG_LEVEL == "default": print("Нет элементов для вывода.")
        return

    # Ordered arrays according to the final path
    ordered_indices = np.array(order, dtype=int)   # indices w.r.t original feats
    ordered_feats = feats[ordered_indices]         # shape (n, D)
    ordered_paths = [paths[i] for i in ordered_indices]

    lines = []
    header = "path\tindex\tdistance"
    lines.append(header)

    # Для каждого элемента посчитать расстояния до соседей по порядку (в пределах границ)
    for pos in range(n):
        cur_feat = ordered_feats[pos]
        left_start = max(0, pos - neighbors)
        left_idxs = list(range(left_start, pos))     # позиции в ordered_* для левой стороны
        right_end = min(n, pos + neighbors + 1)
        right_idxs = list(range(pos + 1, right_end)) # позиции в ordered_* для правой стороны

        neighbor_items = []
        # сначала левое (в порядке от ближнего к дальнему)
        for idx in reversed(left_idxs):  # reversed чтобы ближайший слева идти первым
            dist = float(np.linalg.norm(cur_feat - ordered_feats[idx]))
            neighbor_items.append(f"{idx}:{dist:.6f}")
        # затем правое (от ближнего к дальнему)
        for idx in right_idxs:
            dist = float(np.linalg.norm(cur_feat - ordered_feats[idx]))
            neighbor_items.append(f"{idx}:{dist:.6f}")

        line = f"{ordered_paths[pos]}\t{pos}\t{','.join(neighbor_items)}"
        lines.append(line)

    output_text = "\n".join(lines)

    if out_file:
        # Сохраняем в файл с utf-8
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        if ARG_LOG_LEVEL == "default": print(f"Saved neighbor list to: {out_file}")
    else:
        if ARG_LOG_LEVEL == "default": print(output_text)


