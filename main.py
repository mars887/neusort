import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from cli import CONFIG, LOGGER, TASK_PLAN, HEAVY_TASKS
from core import run_sorting_pipeline, run_search_pipeline, run_clustering_pipeline
from database import list_database_files, move_database_entries

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

    # Validate input folder only if we are going to run a heavy task
    if any(t in HEAVY_TASKS for t in TASK_PLAN):
        if not os.path.isdir(CONFIG.files.src_folder):
            LOGGER.error(f"Source folder {CONFIG.files.src_folder} not found.")
            os.makedirs(CONFIG.files.src_folder, exist_ok=True)
            sys.exit(1)

    # Execute tasks in the order they were requested
    for t in TASK_PLAN:
        if t == "--move_db" and CONFIG.misc.move_db:
            old_root, new_root = CONFIG.misc.move_db
            moved, skipped, missing = move_database_entries(DB_FILE, old_root, new_root)
            LOGGER.info(f"move_db summary -> moved: {moved}, skipped: {skipped}, missing: {missing}")
            continue

        if t == "--list_objects" and CONFIG.misc.list_objects:
            paths = list_database_files(DB_FILE)
            if paths:
                for item in paths:
                    print(item)
            else:
                LOGGER.info("No files found in the database.")
            continue

        if t == "--cluster" and CONFIG.clustering.enabled:
            run_clustering_pipeline(CONFIG, DB_FILE, INDEX_FILE)
            continue

        if t == "--find" and CONFIG.search.find:
            run_search_pipeline(CONFIG, DB_FILE, INDEX_FILE)
            continue

        if t == "--sorting":
            run_sorting_pipeline(CONFIG, DB_FILE, INDEX_FILE)
