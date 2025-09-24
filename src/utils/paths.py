from pathlib import Path


TRAINED_MODELS_DIR = Path("output/models")
DEEPSPEED_CONFIG_DIR = Path("src/training/deepspeed_configs/stage-2-offloading-warmup-cosine-lr.json")

SPIDER_INPUT_DIR = Path("input/qpl/spider")
DB_SCHEMAS_JSON_PATH = SPIDER_INPUT_DIR / "db_schemas.json"
DB_CONTENT_PATH = SPIDER_INPUT_DIR / "db_content.json"
DB_PROFILES_PATH = SPIDER_INPUT_DIR / "db_metadata.json"

SPIDER_DB_CREATION_DIR = SPIDER_INPUT_DIR / "db_creation"
DB_CREATION_PICKLE_DATA_PATH = SPIDER_DB_CREATION_DIR / "data_to_insert_no_alters.pkl"
DB_CREATION_SCHEMAS_DDL_DIR = SPIDER_DB_CREATION_DIR / "schemas"
DB_CREATION_TABLES_SORTED_PATH = SPIDER_DB_CREATION_DIR / "tables-sorted.json"

MANUALLY_LABLED_TYPES_DATASETS = Path("input/qpl/types/manually_labled")
AUTOMATICALLY_LABLED_TYPES_DATASETS = Path("input/qpl/types/automatically_labled")

OUTPUT_DIR = Path("output/qpl")
TYPES_OUTPUT_DIR = OUTPUT_DIR / "types"
ABLATION_OUTPUT_DIR = OUTPUT_DIR / "ablation"
ABLATION_DECOMPOSER_OUTPUT_DIR = ABLATION_OUTPUT_DIR / "decomposer"
ABLATION_COMPLETER_OUTPUT_DIR = ABLATION_OUTPUT_DIR / "completer"