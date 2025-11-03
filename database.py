

# ---------------------------------------------------------------------------- #
#                     5) Кэширование признаков в базе данных                     #
# ---------------------------------------------------------------------------- #
import os
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm.auto import tqdm
from features import extract_feature, Config


def canonical_path(path: str) -> str:
    """Return a normalized representation of a path for comparisons."""
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))
from cli import LOGGER
from models import load_model


def process_and_cache_features(db_file, config: Config):
    """Config
    Универсальная функция для создания и обновления базы данных с признаками.

    Сканирует папку, находит файлы, отсутствующие в базе, обрабатывает их
    пакетами и сразу сохраняет в БД пакетами. Это гарантирует низкое потребление
    оперативной памяти даже при первом запуске на миллионах файлов.

    Args:
        db_file (str): Путь к файлу базы данных SQLite.
        src_folder (str): Путь к папке с исходными изображениями.
        model: Загруженная модель PyTorch.
        hook (dict): Словарь для извлечения признаков из хука модели.
        more_scan (bool): Флаг использования избыточного сканирования.
        batch_size (int): Количество файлов для обработки перед сохранением в БД.
    """
    LOGGER.info(f"Проверка и обновление кэша признаков в файле: {db_file}")
    conn = create_connection(db_file)
    if conn is None:
        LOGGER.error(" ! Не удалось создать соединение с базой данных.")
        return

    try:
        # Убедимся, что таблица существует
        create_table(conn)
        cursor = conn.cursor()

        # 1. Получаем список уже обработанных файлов из БД
        cursor.execute('SELECT filename FROM features')
        # Используем полный путь для корректного сравнения
        existing_rows = [row[0] for row in cursor.fetchall()]
        existing_paths_in_db = {canonical_path(path) for path in existing_rows}
        num_existing = len(existing_rows)
        if num_existing > 0:
            LOGGER.info(f"База данных найдена. В ней уже есть {num_existing} записей.")
        # 2. Получаем актуальный список файлов на диске
        supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")
        current_paths_map = {}
        for fn in os.listdir(config.files.src_folder):
            if not fn.lower().endswith(supported_exts):
                continue
            full_path = os.path.abspath(os.path.join(config.files.src_folder, fn))
            current_paths_map[canonical_path(full_path)] = full_path
        current_paths_on_disk = set(current_paths_map.keys())

        # 3. Находим файлы, которые нужно обработать
        paths_to_process_keys = sorted(list(current_paths_on_disk - existing_paths_in_db))
        paths_to_process = [current_paths_map[key] for key in paths_to_process_keys]
        if not paths_to_process:
            LOGGER.info("База данных актуальна. Новых файлов для обработки не найдено.")
            return
        
        model, hook = load_model(config.model.model_name)

        LOGGER.info(f"Найдено {len(paths_to_process)} новых изображений для обработки.")
        
        # 4. Batch processing of image paths with optional threading
        max_workers_global = max(1, getattr(config.model, "feature_workers", 1))
        use_gpu = torch.cuda.is_available() and not config.model.use_cpu
        if use_gpu and max_workers_global > 1:
            LOGGER.info("GPU inference detected; forcing feature extraction to run sequentially to avoid CUDA thread contention.")
            max_workers_global = 1

        def _process_path(full_path):
            try:
                feat = extract_feature(full_path, model, hook, config)
                if feat is None:
                    return None
                safe_path = os.path.normpath(full_path)
                feature_blob = sqlite3.Binary(feat.tobytes())
                return safe_path, feature_blob
            except Exception as e:
                LOGGER.error(f" ! Error processing image {full_path}: {e}")
                return None

        for i in range(0, len(paths_to_process), config.model.batch_size):
            batch_paths = paths_to_process[i:i + config.model.batch_size]
            batch_feats_data = []
            desc_text = f"Batch {i//config.model.batch_size + 1}/{(len(paths_to_process) + config.model.batch_size - 1)//config.model.batch_size}"
            max_workers = min(max_workers_global, len(batch_paths))

            if max_workers <= 1:
                for full_path in tqdm(batch_paths, desc=desc_text):
                    result = _process_path(full_path)
                    if result is not None:
                        batch_feats_data.append(result)

            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_process_path, full_path): full_path for full_path in batch_paths}
                    for future in tqdm(as_completed(futures), total=len(batch_paths), desc=desc_text):
                        full_path = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            LOGGER.error(f" ! Error processing image {full_path}: {e}")
                            continue
                        if result is not None:
                            batch_feats_data.append(result)


            if batch_feats_data:
                try:
                    before_changes = conn.total_changes
                    cursor.executemany('''
                        INSERT OR REPLACE INTO features (filename, features) VALUES (?, ?)
                    ''', batch_feats_data)
                    conn.commit()
                    written = conn.total_changes - before_changes
                    LOGGER.info(f"Saved {written} records (processed {len(batch_feats_data)} images).")
                    existing_paths_in_db.update(canonical_path(item[0]) for item in batch_feats_data)
                except Exception as e:
                    LOGGER.error(f" ! Failed to write batch into database: {e}")
                    conn.rollback()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finally:
        if conn:
            conn.close()


def create_connection(db_file):
    """ Создает соединение с базой данных SQLite """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        LOGGER.error(e)
    return conn

def create_table(conn):
    """ Создает таблицу для хранения признаков """
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                features BLOB
            )
        ''')
    except sqlite3.Error as e:
        LOGGER.error(e)



def load_features_from_db(db_file):
    """
    Загружает признаки из базы данных.
    Проверяет, существуют ли файлы на диске, и пропускает записи об удаленных файлах.
    Если БД нет или она пуста — возвращает (None, None).
    """
    # Быстрая проверка, есть ли вообще файл БД
    if not os.path.exists(db_file):
        return None, None

    conn = create_connection(db_file)
    if conn is None:
        return None, None

    cursor = conn.cursor()
    try:
        # Извлекаем все записи из базы
        cursor.execute('SELECT filename, features FROM features ORDER BY id')
    except sqlite3.OperationalError:
        # Таблицы 'features' может не существовать, если база создана, но пуста
        conn.close()
        return None, None

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None, None

    # --- НОВАЯ ЛОГИКА ПРОВЕРКИ СУЩЕСТВОВАНИЯ ФАЙЛОВ ---
    
    paths = []
    feats_list = []
    deleted_count = 0

    LOGGER.info("Проверка актуальности записей в базе данных...")
    for filename, features_blob in rows:
        # Проверяем, что файл, указанный в базе, все еще существует
        if os.path.exists(filename):
            paths.append(filename)
            feats_list.append(np.frombuffer(features_blob, dtype=np.float32))
        else:
            # Файла нет, просто пропускаем эту запись, не трогая базу
            deleted_count += 1

    if deleted_count > 0:
        LOGGER.info(f"Пропущено {deleted_count} записей для удаленных файлов.")

    # Если после фильтрации ничего не осталось (например, удалили все картинки)
    if not paths:
        return None, None

    # Возвращаем только данные для существующих файлов
    return np.vstack(feats_list), paths


def list_database_files(db_file):
    """Return the list of filenames stored in the features table ordered by insertion id."""
    if not os.path.exists(db_file):
        LOGGER.error(f"Database file not found: {db_file}")
        return []

    conn = create_connection(db_file)
    if conn is None:
        return []

    cursor = conn.cursor()
    try:
        cursor.execute('SELECT filename FROM features ORDER BY id')
    except sqlite3.OperationalError:
        LOGGER.error("The features table is missing in the database.")
        conn.close()
        return []

    rows = [row[0] for row in cursor.fetchall()]
    conn.close()
    return rows


def move_database_entries(db_file, old_root, new_root):
    """
    Move files referenced in the database from old_root to new_root and update stored paths.

    Returns a tuple with counts: (moved, skipped, missing).
    """
    if not os.path.exists(db_file):
        LOGGER.error(f"Database file not found: {db_file}")
        return 0, 0, 0

    conn = create_connection(db_file)
    if conn is None:
        return 0, 0, 0

    old_root_abs = os.path.abspath(old_root)
    new_root_abs = os.path.abspath(new_root)
    old_root_norm = canonical_path(old_root_abs)

    if not os.path.isdir(old_root_abs):
        LOGGER.error(f"Source directory does not exist: {old_root_abs}")
        conn.close()
        return 0, 0, 0

    os.makedirs(new_root_abs, exist_ok=True)

    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, filename FROM features ORDER BY id')
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        LOGGER.error("The features table is missing in the database.")
        conn.close()
        return 0, 0, 0

    moved = skipped = missing = 0
    for row_id, stored_path in rows:
        if not stored_path:
            continue

        current_abs = os.path.abspath(stored_path)
        current_norm = canonical_path(current_abs)

        try:
            common_root = os.path.commonpath([old_root_norm, current_norm])
        except ValueError:
            continue

        if common_root != old_root_norm:
            continue

        if not os.path.exists(current_abs):
            LOGGER.warning(f"File is missing on disk, skipping: {current_abs}")
            missing += 1
            continue

        try:
            relative_path = os.path.relpath(current_abs, old_root_abs)
        except ValueError:
            LOGGER.warning(f"Failed to compute relative path for: {current_abs}")
            skipped += 1
            continue

        destination_abs = os.path.abspath(os.path.join(new_root_abs, relative_path))
        destination_dir = os.path.dirname(destination_abs)
        os.makedirs(destination_dir, exist_ok=True)

        if os.path.exists(destination_abs):
            LOGGER.warning(f"Destination already exists, skipping move: {destination_abs}")
            skipped += 1
            continue

        try:
            shutil.move(current_abs, destination_abs)
        except (OSError, shutil.Error) as exc:
            LOGGER.error(f"Failed to move '{current_abs}' to '{destination_abs}': {exc}")
            skipped += 1
            continue

        cursor.execute('UPDATE features SET filename = ? WHERE id = ?', (destination_abs, row_id))
        moved += 1

    conn.commit()
    conn.close()
    return moved, skipped, missing

