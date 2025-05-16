from .qpl_to_cte import flat_qpl_to_cte
from processors.qpl import QPLDecomposerProcessor, QPLComposerProcessor


def text_to_qpl(example: dict, db_id: str, schema: str) -> str:
    decomposer_processor = QPLDecomposerProcessor()
    composer_processor = QPLComposerProcessor()




