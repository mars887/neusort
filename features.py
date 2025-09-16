# ---------------------------------------------------------------------------- #
#                         4) Препроцессинг + извлечение                         #
# ---------------------------------------------------------------------------- #
# Стандартный препроцессор для быстрого режима
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from PIL import Image

from cli import DEVICE, LOGGER
from config import Config
from models import MODEL_CONFIGS
Image.MAX_IMAGE_PIXELS = None  # Снимает ограничение на размер изображений
from clip_manager import CLIP_PROCESSOR_MANAGER


def extract_feature(path, model, hook_blob, config: Config):
    """
    Извлекает вектор признаков из одного изображения.
    Поддерживает стандартный и избыточный (more_scan) режимы.
    Поддерживает special-case для CLIP (MODEL_NAME начинается с "clip_"),
    использует глобальную переменную CLIP_PROCESSOR, если она объявлена.
    Возвращает numpy массив float32 (L2-нормализованный).
    """
    img = Image.open(path).convert("RGB")

    # --- выбор размера / препроцессинга в зависимости от модели ---
    is_clip = config.model.model_name.startswith("clip_")
    if is_clip:
        # Рекомендуемый размер для clip_vitl14@336 — 336; можно поменять в MODEL_CONFIGS
        pix_dim = MODEL_CONFIGS.get(config.model.model_name, {}).get("input_size", 336)
        clip_processor = CLIP_PROCESSOR_MANAGER.processor
        if clip_processor is None:
            LOGGER.error("CLIP_PROCESSOR не инициализирован, CLIP-модель не готова к использованию")
    else:
        if config.model.model_name in ("convnext_small", "mobilenet_v3_small", "mobilenet_v3_large"):
            pix_dim = 224
        elif config.model.model_name in ("regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf", "regnet_y_8gf"):
            pix_dim = MODEL_CONFIGS.get(config.model.model_name, {}).get("input_size", 232)
        elif config.model.model_name in ("regnet_y_16gf", "regnet_y_32gf", "regnet_y_128gf"):
            pix_dim = MODEL_CONFIGS.get(config.model.model_name, {}).get("input_size", 384)
        elif config.model.model_name in ("efficientnet_v2_m","efficientnet_v2_l","efficientnet_v2_s"):
            pix_dim = MODEL_CONFIGS.get(config.model.model_name, {}).get("input_size", 480)
        else:
            raise ValueError(f"Неизвестная модель: {config.model.model_name}")

    # --- вспомогательная функция: взять feat после forward (hook_blob должен быть заполнен) ---
    def fetch_feat_from_hook():
        if "feat" not in hook_blob:
            raise RuntimeError("hook_blob не заполнился — hook не сработал")
        t = hook_blob["feat"]
        # иногда hook сохраняет shape (1, D) либо (D,), приводим к 1D numpy float32
        if isinstance(t, torch.Tensor):
            arr = t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            arr = np.asarray(t).reshape(-1).astype(np.float32)
        return arr

    # --- single crop режим ---
    if not config.model.more_scan:
        if is_clip:
            # Используем CLIPProcessor если доступен — он сделает корректную нормализацию и ресайз
            if clip_processor is not None:
                inputs = clip_processor(images=img, return_tensors="pt")  # uses model's expected size
                pixel_values = inputs["pixel_values"].to(DEVICE)
                with torch.no_grad():
                    _ = model(pixel_values)
                feat = fetch_feat_from_hook()
            else:
                # fallback: вручную применяем CLIP-нормализацию (mean/std) и размер pix_dim
                preproc = transforms.Compose([
                    transforms.Resize(pix_dim, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(pix_dim),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711]),
                ])
                x = preproc(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    _ = model(x)
                feat = fetch_feat_from_hook()
        else:
            # прежний путь для torchvision-моделей
            preproc = transforms.Compose([
                transforms.Resize(pix_dim, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(pix_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            x = preproc(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _ = model(x)
            feat = fetch_feat_from_hook()

        # L2-normalize (рекомендуется для стабильного поиска; можно убрать при необходимости)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    # ------------------ more_scan mode (несколько кропов) ------------------
    w, h = img.size
    if h == 0 or w == 0:
        raise ValueError("Изображение имеет нулевой размер")

    collected_feats = []

    # Для CLIP используем processor на каждом кропе (если есть), иначе manual transforms
    if is_clip and clip_processor is None:
        final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        resize_after_crop = transforms.Resize((pix_dim, pix_dim), interpolation=InterpolationMode.BICUBIC)
    elif not is_clip:
        final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        resize_after_crop = transforms.Resize((pix_dim, pix_dim), interpolation=InterpolationMode.BICUBIC)
    else:
        # Если is_clip and clip_processor exists, мы будем применять clip_processor к каждому PIL-кропу,
        # поэтому explicit final_transform не нужен.
        final_transform = None
        resize_after_crop = None

    def get_feat_from_crop(pil_img_crop):
        """Принимает PIL Image (кроп)"""
        if is_clip and clip_processor is not None:
            inputs = clip_processor(images=pil_img_crop, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(DEVICE)
            with torch.no_grad():
                _ = model(pixel_values)
            f = fetch_feat_from_hook()
        else:
            if resize_after_crop is not None:
                pil_img_crop = resize_after_crop(pil_img_crop)
            x = final_transform(pil_img_crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _ = model(x)
            f = fetch_feat_from_hook()
        return f

    # 1. Вычисляем "неквадратность"
    aspect_ratio = max(w, h) / min(w, h)
    is_horizontal = w >= h

    # 2. Вычисляем набор кропов (как у вас)
    crops_to_process = []
    if aspect_ratio < 1.3:
        crops_to_process.append('center')
    elif aspect_ratio < 1.7:
        if is_horizontal:
            crops_to_process.extend(['top_left', 'bottom_right'])
        else:
            crops_to_process.extend(['top_center', 'bottom_center'])
    else:
        crops_to_process.append('center')
        if is_horizontal:
            crops_to_process.extend(['top_left', 'bottom_right'])
        else:
            crops_to_process.extend(['top_center', 'bottom_center'])

    crop_size = min(w, h)

    for crop_type in crops_to_process:
        if crop_type == 'center':
            cropped_img = transforms.functional.center_crop(img, (crop_size, crop_size))
        elif crop_type == 'top_left':
            cropped_img = transforms.functional.crop(img, top=0, left=0, height=crop_size, width=crop_size)
        elif crop_type == 'bottom_right':
            cropped_img = transforms.functional.crop(img, top=h - crop_size, left=w - crop_size, height=crop_size, width=crop_size)
        elif crop_type == 'top_center':
            left = (w - crop_size) // 2
            cropped_img = transforms.functional.crop(img, top=0, left=left, height=crop_size, width=crop_size)
        elif crop_type == 'bottom_center':
            left = (w - crop_size) // 2
            cropped_img = transforms.functional.crop(img, top=h - crop_size, left=left, height=crop_size, width=crop_size)
        else:
            continue

        fv = get_feat_from_crop(cropped_img)
        collected_feats.append(fv)

    if not collected_feats:
        raise ValueError("Не удалось извлечь признаки в режиме more_scan")

    final_feat = np.mean(np.stack(collected_feats, axis=0), axis=0).astype(np.float32)

    # L2-нормировка итогового вектора
    norm = np.linalg.norm(final_feat)
    if norm > 0:
        final_feat = final_feat / norm

    return final_feat
