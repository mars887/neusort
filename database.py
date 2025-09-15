

# ---------------------------------------------------------------------------- #
#                     5) Кэширование признаков в базе данных                     #
# ---------------------------------------------------------------------------- #
import os
import sqlite3

import numpy as np
import torch
from tqdm.auto import tqdm 
from features import extract_feature, Config
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
        existing_paths_in_db = {row[0] for row in cursor.fetchall()}
        num_existing = len(existing_paths_in_db)
        if num_existing > 0:
            LOGGER.info(f"База данных найдена. В ней уже есть {num_existing} записей.")
        # 2. Получаем актуальный список файлов на диске
        supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")
        current_paths_on_disk = {
            os.path.join(config.files.src_folder, fn) for fn in os.listdir(config.files.src_folder)
            if fn.lower().endswith(supported_exts)
        }

        # 3. Находим файлы, которые нужно обработать
        paths_to_process = sorted(list(current_paths_on_disk - existing_paths_in_db))

        if not paths_to_process:
            LOGGER.info("База данных актуальна. Новых файлов для обработки не найдено.")
            return
        
        model, hook = load_model(config.model.model_name)

        LOGGER.info(f"Найдено {len(paths_to_process)} новых изображений для обработки.")
        
        # 4. Обрабатываем новые файлы пакетами
        for i in range(0, len(paths_to_process), config.model.batch_size):
            batch_paths = paths_to_process[i:i + config.model.batch_size]
            batch_feats_data = []
            
            desc_text = f"Обработка пакета {i//config.model.batch_size + 1}/{(len(paths_to_process) + config.model.batch_size - 1)//config.model.batch_size}"
            
            for full_path in tqdm(batch_paths, desc=desc_text):
                try:
                    feat = extract_feature(full_path, model, hook, config)
                    if feat is not None:
                        # Готовим данные для вставки в БД
                        safe_path = full_path.encode('utf-8', errors='replace').decode('utf-8')
                        feature_blob = sqlite3.Binary(feat.tobytes())
                        batch_feats_data.append((safe_path, feature_blob))
                except Exception as e:
                    LOGGER.error(f" ! Ошибка при обработке файла {full_path}: {e}")

            # 5. Сохраняем пакет в базу данных
            if batch_feats_data:
                try:
                    cursor.executemany('''
                        INSERT OR IGNORE INTO features (filename, features) VALUES (?, ?)
                    ''', batch_feats_data)
                    conn.commit()
                    LOGGER.info(f"Успешно сохранено {len(batch_feats_data)} новых признаков в БД.")
                except Exception as e:
                    LOGGER.error(f" ! Ошибка при сохранении пакета в базу данных: {e}")
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

