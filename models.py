CLIP_PROCESSOR = None

MODEL_CONFIGS = {
    "mobilenet_v3_small": {"loader": lambda: __import__("torchvision.models", fromlist=["mobilenet_v3_small"]).mobilenet_v3_small, "weights": "IMAGENET1K_V1", "hook_target": ("classifier", 3), "feat_dim": 1280},
    "mobilenet_v3_large": {"loader": lambda: __import__("torchvision.models", fromlist=["mobilenet_v3_large"]).mobilenet_v3_large, "weights": "IMAGENET1K_V1", "hook_target": ("classifier", 3), "feat_dim": 1280},
    "convnext_small": {"loader": lambda: __import__("torchvision.models", fromlist=["convnext_small"]).convnext_small, "weights": "IMAGENET1K_V1", "hook_target": ("classifier", 2), "feat_dim": 768},
    "regnet_y_400mf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_400mf"]).regnet_y_400mf,
        "weights": "IMAGENET1K_V2",
        "hook_target": ("fc", None),
        "feat_dim": None,
        "input_size": 232
    },
    "regnet_y_800mf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_800mf"]).regnet_y_800mf,
        "weights": "IMAGENET1K_V2",
        "hook_target": ("fc", None),
        "feat_dim": None,
        "input_size": 232
    },
    "regnet_y_1_6gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_1_6gf"]).regnet_y_1_6gf,
        "weights": "IMAGENET1K_V2",
        "hook_target": ("fc", None),
        "feat_dim": None,
        "input_size": 232
    },
    "regnet_y_3_2gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_3_2gf"]).regnet_y_3_2gf,
        "weights": "IMAGENET1K_V2",
        "hook_target": ("fc", None),
        "feat_dim": 1512,
        "input_size": 232
    },
    "regnet_y_8gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_8gf"]).regnet_y_8gf,
        "weights": "IMAGENET1K_V2", 
        "hook_target": ("fc", None),
        "feat_dim": 2016,
        "input_size": 232
    },
    "regnet_y_16gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_16gf"]).regnet_y_16gf,
        "weights": "IMAGENET1K_SWAG_E2E_V1",
        "hook_target": ("fc", None),
        "feat_dim": 2592, 
        "input_size": 384
    },
    "regnet_y_32gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_32gf"]).regnet_y_32gf,
        "weights": "IMAGENET1K_SWAG_E2E_V1",
        "hook_target": ("fc", None),
        "feat_dim": 3712,
        "input_size": 384
    },
    "regnet_y_128gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_128gf"]).regnet_y_128gf,
        "weights": "IMAGENET1K_SWAG_E2E_V1",
        "hook_target": ("fc", None),
        "feat_dim": 7392,
        "input_size": 384
    },
    "efficientnet_v2_s": {
        "loader": lambda: __import__("torchvision.models", fromlist=["efficientnet_v2_s"]).efficientnet_v2_s,
        "weights": "IMAGENET1K_V1", 
        "hook_target": ("auto", None),
        "input_size": 384
    },
    "efficientnet_v2_m": {
        "loader": lambda: __import__("torchvision.models", fromlist=["efficientnet_v2_m"]).efficientnet_v2_m,
        "weights": "IMAGENET1K_V1",  
        "hook_target": ("auto", None),
        "input_size": 480
    },
    "efficientnet_v2_l": {
        "loader": lambda: __import__("torchvision.models", fromlist=["efficientnet_v2_l"]).efficientnet_v2_l,
        "weights": "IMAGENET1K_V1",
        "hook_target": ("auto", None),
        "input_size": 480
    },
    "clip_vit_large": {
        "loader": None,
        "weights": "openai/clip-vit-large-patch14-336",
        "hook_target": (None, None),
        "feat_dim": None,
        "backend": "transformers",       # использовать transformers (OpenAI)
        "input_size": 336,
    },
    "clip_vit_liaon": {
        "loader": None,
        "weights": "laion2b_s32b_b79k",  
        "hook_target": (None, None),
        "feat_dim": None,
        "backend": "open_clip",
        "openclip_model": "ViT-H-14",
    },

    "clip_vit_liaon_mega": {
        "loader": None,
        "weights": "laion2b_s39b_b160k", 
        "hook_target": (None, None),
        "feat_dim": None,
        "backend": "open_clip",
        "openclip_model": "ViT-bigG-14",
    },
} 

# ---------------------------------------------------------------------------- #
#                             3) Загрузка модели                                 #
# ---------------------------------------------------------------------------- #
import os
import torch
import torch.nn as nn
from PIL import Image

from cli import ARG_LOG_LEVEL, DEVICE

def find_final_linear_module(model):
    """
    Находит последний nn.Linear в модели и возвращает (parent_module, attr_name_or_index)
    parent_module — модуль, у которого нужно взять атрибут/элемент,
    attr_name_or_index — имя атрибута или индекс в nn.Sequential.
    Возвращает (None, None) если не найдено.
    """
    last_linear = None
    last_linear_name = None
    # используем named_modules чтобы получить полные имена
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module
            last_linear_name = name

    if last_linear is None:
        return None, None

    # last_linear_name вроде "classifier.1" или "_fc"
    parts = last_linear_name.split('.')
    if len(parts) == 1:
        # атрибут верхнего уровня, например "_fc"
        parent_name = parts[0]
        parent = model
        return parent_name, None  # загрузчик обработает как getattr(model, parent_name)
    else:
        # parent = getattr(model, "a.b...") where last part is attribute/index
        parent_attr = '.'.join(parts[:-1])
        last_part = parts[-1]
        # получим объект parent
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        # если last_part — число => index
        try:
            idx = int(last_part)
            return parent_attr, idx
        except Exception:
            return parent_attr, last_part
        
        
def load_model(model_name):

    # --- special-case: CLIP models via transformers ---
    if model_name.startswith("clip_"):
        cfg = MODEL_CONFIGS[model_name]
        model_id = cfg["weights"]
        backend = cfg.get("backend", "transformers")
        if ARG_LOG_LEVEL == "default": print(f"Loading CLIP model ({model_name}) with backend={backend}: {model_id} ...")

        hook_blob = {}

        if backend == "open_clip":
            # open_clip backend (recommended for LAION weights / bigG/H variants)
            try:
                import open_clip
            except Exception as e:
                raise RuntimeError("open_clip not installed. pip install open-clip-torch") from e

            oc_model_name = cfg.get("openclip_model")
            if oc_model_name is None:
                raise ValueError("MODEL_CONFIGS for open_clip must include 'openclip_model' key (e.g. 'ViT-bigG-14').")

            # create model + preprocess
            model_oc, _, preprocess = open_clip.create_model_and_transforms(oc_model_name, pretrained=cfg["weights"])
            model_oc.to(DEVICE).eval()

            # wrapper so our code can call model(pixel_values) like before
            class OCWrapper(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner
                def forward(self, pixel_values):
                    # pixel_values maybe dict {"pixel_values": tensor} or tensor directly
                    if isinstance(pixel_values, dict):
                        x = pixel_values["pixel_values"]
                    else:
                        x = pixel_values
                    # open_clip: use encode_image
                    return self.inner.encode_image(x)

            model = OCWrapper(model_oc)

            # ------------------ вместо старого oc_processor определим этот ------------------
            def oc_processor(*args, **kwargs):
                """
                Совместимый processor для open_clip: принимает (images=pil_or_list, return_tensors="pt")
                Возвращает dict с ключом "pixel_values" => тензор shape (B, C, H, W) на CPU.
                """
                # Поддерживаем совместимый интерфейс: images может быть в kwargs либо первым аргументом
                if "images" in kwargs:
                    imgs = kwargs["images"]
                elif len(args) >= 1:
                    imgs = args[0]
                else:
                    raise TypeError("oc_processor требует аргумент images= или позиционный параметр с PIL image/списком")
            
                # Если передали один PIL — оборачиваем в список
                single = False
                if isinstance(imgs, (Image.Image,)):
                    imgs = [imgs]
                    single = True
                elif isinstance(imgs, (list, tuple)):
                    imgs = list(imgs)
                    single = False
                else:
                    # возможно передали tensor уже — попробуем использовать как есть
                    # в большинстве случаев это не нужно; бросим ошибку чтобы не молча ломать
                    raise TypeError("oc_processor ожидает PIL Image или список PIL Image в 'images'")
            
                # применяем preprocess к каждому PIL и собираем батч
                tensors = []
                for im in imgs:
                    t = preprocess(im)  # preprocess возвращает tensor C,H,W
                    if not torch.is_tensor(t):
                        t = torch.tensor(t)
                    tensors.append(t)
            
                batch = torch.stack(tensors, dim=0)  # shape (B, C, H, W)
            
                # если просят return_tensors=="pt" — возвращаем torch.Tensor, иначе всё равно возвращаем tensor
                # Приводим к device CPU; модель ожидает перемещение на DEVICE уже в extract_feature
                return {"pixel_values": batch}
            # -------------------------------------------------------------------------------

            global CLIP_PROCESSOR
            CLIP_PROCESSOR = oc_processor

            # hook for collecting features
            def _hook(module, input, output):
                # output is tensor (B,D)
                hook_blob["feat"] = output.detach().cpu().clone()
            model.register_forward_hook(_hook)

            # determine feat_dim (use preprocess size if possible)
            with torch.no_grad():
                # build dummy using preprocess size if exists
                try:
                    tmp_size = getattr(preprocess, "size", None)
                    if tmp_size is None:
                        tmp_size = 336
                    dummy = preprocess(Image.new("RGB", (tmp_size, tmp_size))).unsqueeze(0).to(DEVICE)
                    out = model({"pixel_values": dummy})
                except Exception:
                    # try generic
                    dummy = torch.zeros(1, 3, 336, 336, device=DEVICE)
                    out = model(dummy)
                feat_dim = int(out.shape[-1])

            MODEL_CONFIGS[model_name]["feat_dim"] = feat_dim
            MODEL_CONFIGS[model_name]["input_size"] = getattr(preprocess, "size", MODEL_CONFIGS[model_name].get("input_size", 336))
            if ARG_LOG_LEVEL == "default": print(f"OpenCLIP loaded. feat_dim={feat_dim}, input_size={MODEL_CONFIGS[model_name]['input_size']}")
            return model, hook_blob

        else:
            # transformers backend (openai weights) — existing code, with small tweaks
            from transformers import CLIPModel, CLIPProcessor
            clip_model = CLIPModel.from_pretrained(model_id)
            clip_model.to(DEVICE).eval()
            processor = CLIPProcessor.from_pretrained(model_id)

            # wrapper that calls get_image_features
            class TFWrapper(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner
                def forward(self, pixel_values):
                    return self.inner.get_image_features(pixel_values=pixel_values)

            model = TFWrapper(clip_model)

            def _hook(module, input, output):
                hook_blob["feat"] = output.detach().cpu().clone()
            model.register_forward_hook(_hook)

            # save processor as global in transformers-compatible form
            # global CLIP_PROCESSOR
            CLIP_PROCESSOR = processor

            # determine feat dim
            with torch.no_grad():
                # try processor size if available, else 224
                try:
                    dummy_img = Image.new("RGB", (cfg.get("input_size", 224), cfg.get("input_size", 224)))
                    inputs = processor(images=dummy_img, return_tensors="pt")
                    pv = inputs["pixel_values"].to(DEVICE)
                    out = model(pv)
                except Exception:
                    dummy = torch.zeros(1, 3, cfg.get("input_size", 224), cfg.get("input_size", 224), device=DEVICE)
                    out = model(dummy)
                feat_dim = int(out.shape[-1])

            MODEL_CONFIGS[model_name]["feat_dim"] = feat_dim
            print(f"Transformers CLIP loaded. feat_dim={feat_dim}")
            return model, hook_blob

    
    cfg = MODEL_CONFIGS[model_name]
    model_constructor = cfg["loader"]()
    weights_str = cfg["weights"]

    local_weights_file = f"{model_name}.pth"
    if os.path.exists(local_weights_file):
        try:
            if ARG_LOG_LEVEL == "default": print(f"Найден локальный файл весов '{local_weights_file}'. Загрузка...")
            sd = torch.load(local_weights_file, map_location="cpu")
            model = model_constructor(weights=None)
            model.load_state_dict(sd)
            if ARG_LOG_LEVEL == "default": print("Локальные веса успешно загружены.")
        except Exception as e:
            if ARG_LOG_LEVEL == "default" or ARG_LOG_LEVEL == "error" :print(f"Ошибка при загрузке локальных весов: {e}. Удаляем файл и скачиваем заново.")
            os.remove(local_weights_file)
            if ARG_LOG_LEVEL == "default": print(f"Загрузка предобученных весов '{weights_str}' для {model_name}...")
            model = model_constructor(weights=weights_str)
            torch.save(model.state_dict(), local_weights_file)
            if ARG_LOG_LEVEL == "default": print(f"Веса скачаны и сохранены в '{local_weights_file}'.")
    else:
        if ARG_LOG_LEVEL == "default": print(f"Локальный файл весов не найден. Загрузка предобученных весов '{weights_str}' для {model_name}...")
        model = model_constructor(weights=weights_str)
        torch.save(model.state_dict(), local_weights_file)
        if ARG_LOG_LEVEL == "default": print(f"Веса скачаны и сохранены в '{local_weights_file}'.")

    model.to(DEVICE).eval()

    # -- Автоопределяем реальную размерность выходного признака (fc.in_features)
    #    это защищает от несовпадения feat_dim между версиями torchvision
    try:
        actual_dim = None
        # обычный случай: model.fc - линейный слой
        if hasattr(model, "fc") and hasattr(model.fc, "in_features"):
            actual_dim = int(model.fc.in_features)
        else:
            # ищем первый Linear (fallback)
            import torch.nn as nn
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    actual_dim = int(m.in_features)
                    break

        if actual_dim is not None:
            # обновляем cfg на лету (чтобы остальной код знал точную feat_dim)
            cfg["feat_dim"] = actual_dim
            if ARG_LOG_LEVEL == "default":
                print(f"Определён feat_dim для {model_name}: {actual_dim}")
    except Exception as e:
        if ARG_LOG_LEVEL in ("default", "error"):
            print(f"Не удалось автоопределить feat_dim: {e}")

    # Вешаем hook
    hook_blob = {}
    
    group, idx_or_name = cfg.get("hook_target", (None, None))
    if group == "auto":
        # автопоиск
        g, idx = find_final_linear_module(model)
        if g is None:
            # fallback — попробуем стандартные имена
            for candidate in ("classifier", "_fc", "head", "fc"):
                if hasattr(model, candidate):
                    g = candidate
                    idx = None
                    break
        if g is None:
            raise RuntimeError("Не удалось найти финальный линейный слой для установки hook'а. Укажите hook_target вручную.")
        group, idx_or_name = g, idx


    # Получаем родительский модуль (например, model.classifier или model.head)
    target_module_parent = getattr(model, group)
    
    # Определяем конечный модуль для hook'а
    if idx_or_name is None:
        # Если имя/индекс не указаны, значит родительский модуль и есть цель
        # (случай для swin_v2_t, где head - это и есть Linear слой)
        target_module = target_module_parent
    elif isinstance(idx_or_name, int):
        # Если это индекс, получаем элемент из родителя (случай для maxvit_t, mobilenet, efficientnet)
        target_module = target_module_parent[idx_or_name]
    else: 
        # Если это имя слоя, как 'fc' (случай для старых моделей, например, ResNet)
        target_module = getattr(target_module_parent, idx_or_name)

    def hook_fn(module, input, output):
        hook_blob["feat"] = input[0].detach().cpu().clone()
    
    target_module.register_forward_hook(hook_fn)

    return model, hook_blob