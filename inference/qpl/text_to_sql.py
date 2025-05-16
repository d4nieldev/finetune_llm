from .qpl_to_cte import flat_qpl_to_cte
from processors.qpl import QPLDecomposerProcessor, QPLComposerProcessor
from typing_extensions import TypedDict
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


class InputExample(TypedDict):
    question: str
    db_id: str


def text_to_qpl(example: InputExample) -> str:
    decomposer_processor = QPLDecomposerProcessor()
    composer_processor = QPLComposerProcessor()


def decompose_nlq(
        example: InputExample, 
        processor: QPLDecomposerProcessor,
        model: AutoPeftModelForCausalLM,
        tokenizer: AutoTokenizer,
    ) -> str:

    chat_template = processor.to_chat_template(example)

