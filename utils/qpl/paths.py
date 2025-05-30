from pathlib import Path


SPIDER_INPUT_DIR = Path("input/qpl/spider")
DB_SCHEMAS_JSON_PATH = SPIDER_INPUT_DIR / "db_schemas.json"
DB_CONTENT_PATH = SPIDER_INPUT_DIR / "db_content.json"

SPIDER_DB_CREATION_DIR = SPIDER_INPUT_DIR / "db_creation"
DB_CREATION_PICKLE_DATA_PATH = SPIDER_DB_CREATION_DIR / "data_to_insert_no_alters.pkl"
DB_CREATION_SCHEMAS_DDL_DIR = SPIDER_DB_CREATION_DIR / "schemas"
DB_CREATION_TABLES_SORTED_PATH = SPIDER_DB_CREATION_DIR / "tables-sorted.json"

TYPES_INPUT_DIR = Path("input/qpl/types")
TYPES_CONCERT_SINGER_PATH = TYPES_INPUT_DIR / "question_types_concert_singer.json"

TYPES_OUTPUT_DIR = Path("output/qpl/types")