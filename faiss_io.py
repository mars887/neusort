# ------------------ Save / Load FAISS index (simple) ------------------
import os
import faiss

from cli import ARG_LOG_LEVEL


def save_faiss_index(index, index_path):
    """
    Сохраняет faiss индекс на диск атомарно.
    Если индекс на GPU — сначала переводит его в CPU.
    """
    try:
        # Если GPU-индекс, попытаться перевести на CPU
        try:
            if faiss.get_num_gpus() > 0 and index.__class__.__name__.startswith('Gpu'):
                index_cpu = faiss.index_gpu_to_cpu(index)
            else:
                index_cpu = index
        except Exception:
            index_cpu = index

        tmp_path = index_path + ".tmp"
        faiss.write_index(index_cpu, tmp_path)
        os.replace(tmp_path, index_path)
    except Exception as e:
        if ARG_LOG_LEVEL == "default" or ARG_LOG_LEVEL == "error": print(f"! Не удалось сохранить FAISS индекс: {e}")
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

def load_faiss_index(index_path, use_gpu=False, gpu_id=0):
    """
    Загружает faiss индекс с диска. Опционально переносит на GPU.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    index = faiss.read_index(index_path)
    if use_gpu:
        try:
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        except Exception as e:
            if ARG_LOG_LEVEL == "default": print(f"  - Предупреждение: не удалось перенести индекс на GPU: {e}")
    return index

