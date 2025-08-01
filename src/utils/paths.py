from pathlib import Path


TRAINED_MODELS_DIR = Path("output/models")
DEEPSPEED_CONFIG = Path("src/training/configs/deepspeed_config_no-offload-warmup-cosine-lr.json")