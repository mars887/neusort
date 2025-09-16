import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


from cli import CONFIG, LOGGER
from core import run_sorting_pipeline, run_search_pipeline

if __name__ == "__main__":

    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    safe_model_name = CONFIG.model.model_name.replace("/", "_")
    scan_suffix = "_more_scan" if CONFIG.model.more_scan else ""
    DB_FILE = f"features_db_{safe_model_name}{scan_suffix}.sqlite"
    INDEX_FILE = CONFIG.files.index_file

    if not os.path.isdir(CONFIG.files.src_folder):
        LOGGER.error(f"Создайте папку {CONFIG.files.src_folder} и положите туда картинки.")
        os.makedirs(CONFIG.files.src_folder, exist_ok=True)
        exit(1)

    if CONFIG.search.find:
        run_search_pipeline(CONFIG, DB_FILE, INDEX_FILE)
    else:
        run_sorting_pipeline(CONFIG, DB_FILE, INDEX_FILE)