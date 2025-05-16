import json
from typing import Tuple, Union, Optional, List, Callable, Iterable, Dict, Any
from typing_extensions import TypedDict
from dataclasses import dataclass, asdict
from enum import Enum

# from .qpl_to_cte import flat_qpl_to_cte
from processors.qpl import QPLDecomposerProcessor, QPLComposerProcessor
from custom_types import ChatTemplate
from utils.generation_utils import to_model_prompt, generate_batch
from utils.lists import flatten, unflatten

from peft import AutoPeftModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class InputExample(TypedDict):
    question: str
    db_id: str


class Operator(Enum):
    SCAN = "Scan"
    AGGREGATE = "Aggregate"
    FILTER = "Filter"
    SORT = "Sort"
    TOPSORT = "TopSort"
    JOIN = "Join"
    EXCEPT = "Except"
    INTERSECT = "Intersect"
    UNION = "Union"


@dataclass
class QPLTree:
    question: str
    op: Operator
    children: Optional[Union[Tuple["QPLTree"], Tuple["QPLTree", "QPLTree"]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "op": self.op.value,
            "children": [child.to_dict() for child in self.children] if self.children else None
        }


def text_to_qpl(examples: List[InputExample], decomposer_model_path: str) -> List[str]:
    decomposer_processor = QPLDecomposerProcessor()
    composer_processor = QPLComposerProcessor()

    decomposer_model = AutoPeftModelForCausalLM.from_pretrained(decomposer_model_path).to("cuda")
    decomposer_tokenizer = AutoTokenizer.from_pretrained(decomposer_model_path)

    docomposer_trees = decompose(
        examples=examples,
        processor=decomposer_processor,
        model=decomposer_model,
        tokenizer=decomposer_tokenizer,
    )

    return docomposer_trees

def decompose(
        examples: List[InputExample],
        processor: QPLDecomposerProcessor,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 16,
        max_new_tokens: int = 512,
    ) -> List[QPLTree]:

    if len(examples) == 0:
        return []

    # Create a QPL tree for each example
    trees = [QPLTree(question=ex['question'], op=None) for ex in examples]

    # Use the decomposer model to generate the questions for the next layer of each QPL tree
    chat_templates = list(map(processor.to_chat_template, examples))
    prompts = list(map(lambda ct: to_model_prompt(tokenizer, ct), chat_templates))
    outputs = generate_batch(
        model=model,
        tokenizer=tokenizer,
        model_prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    # Parse the outputs and create child QPL trees
    children_examples = []
    for example, tree, output in zip(examples, trees, outputs):
        lines = output.split("\n")
        tree.op = Operator(lines[0].strip())
        children_examples.append([InputExample(question=l.strip(), db_id=example['db_id']) for l in lines[1:]])

    # Generate the trees of the children questions
    flat_children_examples, lengths = flatten(children_examples)
    flat_children_trees = decompose(
        examples=flat_children_examples,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    children_trees = unflatten(flat_children_trees, lengths)

    # Assign the children trees to the parent trees
    for tree, children in zip(trees, children_trees):
        if len(children) == 0:
            tree.children = None
        elif len(children) == 1:
            tree.children = (children[0],)
        else:
            tree.children = tuple(children)
    
    return trees


if __name__ == "__main__":
    example = InputExample(question="What is the code of airport that has the highest number of flights?", db_id="flight_2")
    decomposer_model_path = "output/gemma-3-4b-it-question_decomposer_ds_train_batch_size=1_gradient_accumulation_steps=1_learning_rate=0.0002_num_train_epochs=2_gradient_checkpointing=False_logging_steps=500_save_steps=5000_random_seed=1_lora=True_r=16_alpha=32_dropout=0.05/checkpoint-20958"
    qpl_tree = text_to_qpl([example], decomposer_model_path)[0]
    print(json.dumps(qpl_tree.to_dict(), indent=2))


