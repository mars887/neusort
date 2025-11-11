MODEL_CONFIGS = {
    "mobilenet_v3_small": {
        "loader": lambda: __import__("torchvision.models", fromlist=["mobilenet_v3_small"]).mobilenet_v3_small,
        "weights": "IMAGENET1K_V1",
        "hook_target": ("classifier", 3),
        "feat_dim": 1280,
        "flops": "~0.117 GF",
    },
    "mobilenet_v3_large": {
        "loader": lambda: __import__("torchvision.models", fromlist=["mobilenet_v3_large"]).mobilenet_v3_large,
        "weights": "IMAGENET1K_V1",
        "hook_target": ("classifier", 3),
        "feat_dim": 1280,
        "flops": "~0.22 GF",
    },
    "convnext_small": {
        "loader": lambda: __import__("torchvision.models", fromlist=["convnext_small"]).convnext_small,
        "weights": "IMAGENET1K_V1",
        "hook_target": ("classifier", 2),
        "feat_dim": 768,
        "flops": "~4.3 GF",
    },
    "regnet_y_400mf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_400mf"]).regnet_y_400mf,
        "weights": "IMAGENET1K_V2",
        "hook_target": ("fc", None),
        "feat_dim": None,
        "input_size": 232,
        "flops": "0.4 GF",
    },
    "regnet_y_800mf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_800mf"]).regnet_y_800mf,
        "weights": "IMAGENET1K_V2",
        "hook_target": ("fc", None),
        "feat_dim": None,
        "input_size": 232,
        "flops": "0.8 GF",
    },
    "regnet_y_1_6gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_1_6gf"]).regnet_y_1_6gf,
        "weights": "IMAGENET1K_V2",
        "hook_target": ("fc", None),
        "feat_dim": None,
        "input_size": 232,
        "flops": "1.6 GF",
    },
    "regnet_y_3_2gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_3_2gf"]).regnet_y_3_2gf,
        "weights": "IMAGENET1K_V2",
        "hook_target": ("fc", None),
        "feat_dim": 1512,
        "input_size": 232,
        "flops": "3.2 GF",
    },
    "regnet_y_8gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_8gf"]).regnet_y_8gf,
        "weights": "IMAGENET1K_V2", 
        "hook_target": ("fc", None),
        "feat_dim": 2016,
        "input_size": 232,
        "flops": "8 GF",
    },
    "regnet_y_16gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_16gf"]).regnet_y_16gf,
        "weights": "IMAGENET1K_SWAG_E2E_V1",
        "hook_target": ("fc", None),
        "feat_dim": 2592, 
        "input_size": 384,
        "flops": "16 GF",
    },
    "regnet_y_32gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_32gf"]).regnet_y_32gf,
        "weights": "IMAGENET1K_SWAG_E2E_V1",
        "hook_target": ("fc", None),
        "feat_dim": 3712,
        "input_size": 384,
        "flops": "32 GF",
    },
    "regnet_y_128gf": {
        "loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_128gf"]).regnet_y_128gf,
        "weights": "IMAGENET1K_SWAG_E2E_V1",
        "hook_target": ("fc", None),
        "feat_dim": 7392,
        "input_size": 384,
        "flops": "128 GF",
    },
    "efficientnet_v2_s": {
        "loader": lambda: __import__("torchvision.models", fromlist=["efficientnet_v2_s"]).efficientnet_v2_s,
        "weights": "IMAGENET1K_V1", 
        "hook_target": ("auto", None),
        "input_size": 384,
        "flops": "~8.42 GF",
    },
    "efficientnet_v2_m": {
        "loader": lambda: __import__("torchvision.models", fromlist=["efficientnet_v2_m"]).efficientnet_v2_m,
        "weights": "IMAGENET1K_V1",  
        "hook_target": ("auto", None),
        "input_size": 480,
        "flops": "~24.74 GF",
    },
    "efficientnet_v2_l": {
        "loader": lambda: __import__("torchvision.models", fromlist=["efficientnet_v2_l"]).efficientnet_v2_l,
        "weights": "IMAGENET1K_V1",
        "hook_target": ("auto", None),
        "input_size": 480,
        "flops": "~56.13 GF",
    },
    "clip_vit_large": {
        "loader": None,
        "weights": "openai/clip-vit-large-patch14-336",
        "hook_target": (None, None),
        "feat_dim": None,
        "backend": "transformers",       # использовать transformers (OpenAI)
        "input_size": 336,
        "flops": "~140 GF",
    },
    "clip_vit_liaon": {
        "loader": None,
        "weights": "laion2b_s32b_b79k",  
        "hook_target": (None, None),
        "feat_dim": None,
        "backend": "open_clip",
        "openclip_model": "ViT-H-14",
        "flops": "depends on input resolution",
    },

    "clip_vit_liaon_mega": {
        "loader": None,
        "weights": "laion2b_s39b_b160k", 
        "hook_target": (None, None),
        "feat_dim": None,
        "backend": "open_clip",
        "openclip_model": "ViT-bigG-14",
        "flops": "depends on input resolution",
    },
    # EVA-02 (ViT-L/14 448) via timm from HF Hub
    "anime_eva02_large": {
        "loader": None,
        "weights": "animetimm/eva02_large_patch14_448.dbv4-full",
        "hook_target": ("auto", None),
        "feat_dim": None,
        "backend": "timm_hf",
        "input_size": 448,
        "normalization": "clip",
        "flops": "~620.9 GF",
    },
    "anime_convnextv2_huge": {
        "weights": "animetimm/convnextv2_huge.dbv4-full",
        "backend": "timm_hf",
        "input_size": 512,
        "normalization": "imagenet",
        "hook_target": ("auto", None),
        "feat_dim": None,
        "flops": "~600.8 GF",
    },
    "anime_swinv2_base_w8": {
        "weights": "animetimm/swinv2_base_window8_256.dbv4-full",
        "backend": "timm_hf",
        "input_size": 448,
        "normalization": "imagenet",
        "hook_target": ("auto", None),
        "feat_dim": None,
        "flops": "~121.6 GF",
    },
    "anime_caformer_b36": {
        "weights": "animetimm/caformer_b36.dbv4-full",
        "backend": "timm_hf",
        "input_size": 384,
        "normalization": "imagenet",
        "hook_target": ("auto", None),
        "feat_dim": None,
        "flops": "~72.2 GF",
    },
    "anime_mobilenetv4_conv_aa_large": {
        "weights": "animetimm/mobilenetv4_conv_aa_large.dbv4-full",
        "backend": "timm_hf",
        "input_size": 448,
        "normalization": "imagenet",
        "hook_target": ("auto", None),
        "feat_dim": None,
        "flops": "~19.2 GF",
    },
} 

# ---------------------------------------------------------------------------- #
#                             3) Загрузка модели                                 #
# ---------------------------------------------------------------------------- #
from PIL import Image

from logger import CustomLogger
from model_factory import create_clip_openclip_model, create_clip_transformers_model, create_torchvision_model
        
LOGGER: CustomLogger = CustomLogger()


def set_model_logger(logger: CustomLogger) -> None:
    global LOGGER
    LOGGER = logger
        
        
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
    else:
        backend = cfg.get("backend")
        if backend == "timm_hf":
            from model_factory import create_timm_hf_model
            model, hook_blob = create_timm_hf_model(model_name, cfg)
            return model, hook_blob

        return create_torchvision_model(model_name, cfg)
