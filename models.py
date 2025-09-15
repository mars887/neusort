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
from PIL import Image

from cli import ARG_LOG_LEVEL, DEVICE
from cli import LOGGER
from model_factory import create_clip_openclip_model, create_clip_transformers_model, create_torchvision_model
        
        
def load_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    model_id = cfg["weights"]
    if model_name.startswith("clip_"):

        backend = cfg.get("backend", "transformers")

        LOGGER.info(f"Loading CLIP model ({model_name}) with backend={backend}: {model_id} ...")

        if backend == "open_clip":
            model, hook_blob = create_clip_openclip_model(model_name, cfg)
        else:
            model, hook_blob = create_clip_transformers_model(model_name, cfg)
        return model, hook_blob
    
    return create_torchvision_model(model_name,cfg)