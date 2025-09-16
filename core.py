# core.py

import torch
from database import process_and_cache_features, load_features_from_db
from sorting import sort_images
from search import handle_search_pipeline
import faiss
from cli import LOGGER

def run_sorting_pipeline(config, db_file, index_file):
    """
    Основной pipeline: обновление БД, загрузка данных, сортировка и сохранение индекса.
    """
    use_gpu_faiss = (not config.model.use_cpu) and torch.cuda.is_available()
    
    # 1. Обновление БД
    # ИСПРАВЛЕНО: config.src_folder -> config.files.src_folder
    process_and_cache_features(db_file, config)
    
    # 2. Загрузка данных
    feats, paths = load_features_from_db(db_file)
    if not paths or feats is None or len(paths) == 0:
        return

    # 3. Построение индекса и запуск сортировки
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
    
    # Запуск сортировки
    LOGGER.info(f"Всего в работе {len(paths)} изображений. Начинаем сортировку...")
    sort_images(feats, paths, config)

def run_search_pipeline(config, db_file, index_file):
    """
    Pipeline для режима поиска (--find).
    """
    # Загрузка данных из БД
    feats, paths = load_features_from_db(db_file)
    if not paths or feats is None or len(paths) == 0:
        LOGGER.error("Нет данных для поиска. Убедитесь, что база данных не пуста.")
        return

    # Передаем управление в search.py
    handle_search_pipeline(config, feats, paths)