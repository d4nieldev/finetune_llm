import json
from typing import Tuple, Union, Optional, List, Callable, Iterable, Dict, Any
from typing_extensions import TypedDict
from dataclasses import dataclass
from enum import Enum

from inference.qpl.qpl_to_cte import flat_qpl_to_cte
from processors.qpl import QPLDecomposerProcessor, QPLComposerProcessor
from utils.generation_utils import to_model_prompt, generate_batch
from utils.lists import flatten, unflatten

from peft import AutoPeftModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class DecomposerExample(TypedDict):
    question: str
    db_id: str


class CompleterExample(TypedDict):
    question: str
    db_id: str
    prefix_qpl: str
    op: str
    parent_question: Optional[str]


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
    db_id: str
    op: Operator = None
    qpl_line: str = None
    line_idx: int = None
    parent: Optional["QPLTree"] = None
    children: Optional[Union[Tuple["QPLTree"], Tuple["QPLTree", "QPLTree"]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "op": self.op.value,
            "prefix_qpl": self.prefix_qpl,
            "qpl": self.qpl,
            "qpl_line": self.qpl_line,
            "children": [child.to_dict() for child in self.children] if self.children else None,
        }

    @property
    def prefix_qpl(self) -> str:
        if self.children is None:
            return ""
        return "\n".join([(child.prefix_qpl + "\n" + child.qpl_line).strip() for child in self.children]).replace("\n", " ; ")
    
    @property
    def qpl(self) -> str:
        return f"{self.prefix_qpl} ; {self.qpl_line}" if self.prefix_qpl else self.qpl_line


def text_to_sql(examples: List[DecomposerExample], decomposer_model_path: str, completer_model_path: str) -> List[str]:
    # Decompose input questions
    decomposer_processor = QPLDecomposerProcessor()
    decomposer_model = AutoPeftModelForCausalLM.from_pretrained(decomposer_model_path).to("cuda")
    decomposer_tokenizer = AutoTokenizer.from_pretrained(decomposer_model_path)
    trees = decompose(
        examples=examples,
        processor=decomposer_processor,
        model=decomposer_model,
        tokenizer=decomposer_tokenizer,
        batch_size=16,
        max_new_tokens=256,
    )
    for tree in trees:
        post_order_index_tree(tree)

    decomposer_model = decomposer_model.to("cpu")

    # complete QPL for trees
    completer_processor = QPLComposerProcessor()
    completer_model = AutoPeftModelForCausalLM.from_pretrained(completer_model_path).to("cuda")
    completer_tokenizer = AutoTokenizer.from_pretrained(completer_model_path)

    complete(
        trees=trees,
        processor=completer_processor,
        model=completer_model,
        tokenizer=completer_tokenizer,
        batch_size=16,
        max_new_tokens=256,
    )

    print(json.dumps([tree.to_dict() for tree in trees], indent=2))

    # Convert QPL to SQL
    flat_qpls = [tree.qpl.split(' ; ') for tree in trees]
    db_ids = [tree.db_id for tree in trees]
    ctes = [flat_qpl_to_cte(flat_qpl, db_id) for flat_qpl, db_id in zip(flat_qpls, db_ids)]

    return ctes
    

def decompose(
        examples: List[DecomposerExample],
        processor: QPLDecomposerProcessor,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 16,
        max_new_tokens: int = 512,
    ) -> List[QPLTree]:

    if len(examples) == 0:
        return []

    # Create a QPL tree for each example
    trees = [QPLTree(question=ex['question'], db_id=ex['db_id']) for ex in examples]

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
        children_examples.append([DecomposerExample(question=l.strip(), db_id=example['db_id']) for l in lines[1:]])

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
        for child in children:
            child.parent = tree
        if len(children) == 0:
            tree.children = None
        elif len(children) == 1:
            tree.children = (children[0],)
        else:
            tree.children = tuple(children)
    
    return trees


def post_order_index_tree(tree: QPLTree, counter: int = 1) -> int:
    if tree.children:
        for child in tree.children:
            counter = post_order_index_tree(child, counter)
    tree.line_idx = counter
    return counter + 1


def complete(
        trees: List[QPLTree],
        processor: QPLComposerProcessor,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 16,
        max_new_tokens: int = 512,
    ) -> None:
    if len(trees) == 0:
        return
    
    # Complete the children before completing the parent
    trees_to_complete = []
    for tree in trees:
        if tree.children is None:
            continue
        for child in tree.children:
            if child.qpl_line is None:
                trees_to_complete.append(child)

    complete(
        trees=trees_to_complete,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    
    # Use the model to generate the QPL for each parent
    examples = [
        CompleterExample(
            question=tree.question,
            db_id=tree.db_id,
            prefix_qpl=tree.prefix_qpl,
            op=tree.op.value,
            parent_question=tree.parent.question if tree.parent else None,
        )
        for tree in trees
    ]
    chat_templates = list(map(processor.to_chat_template, examples))
    prompts = list(map(lambda ct: to_model_prompt(tokenizer, ct), chat_templates))
    outputs = generate_batch(
        model=model,
        tokenizer=tokenizer,
        model_prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    # Parse the outputs and assign the QPL lines to the trees
    for tree, output in zip(trees, outputs):
        completion = output.split('\n')[0].strip()
        tree.qpl_line = f"#{tree.line_idx} = {tree.op.value} {completion}"


if __name__ == "__main__":
    example = DecomposerExample(question="What is the code of airport that has the highest number of flights?", db_id="flight_2")
    decomposer_model_path = "output/gemma-3-4b-it-question_decomposer_ds_train_batch_size=1_gradient_accumulation_steps=1_learning_rate=0.0002_num_train_epochs=2_gradient_checkpointing=False_logging_steps=500_save_steps=5000_random_seed=1_lora=True_r=16_alpha=32_dropout=0.05/checkpoint-20958"
    completer_model_path = "output/erbz0056_gemma-3-4b-it-qpl_composer_train_batch_size=1_gradient_accumulation_steps=8_learning_rate=0.0002_num_train_epochs=4_gradient_checkpointing=True_logging_steps=0.05_save_steps=0.5_random_seed=1_lora=True_r=16_alpha=32_dropout=0.05/checkpoint-5316"
    sql = text_to_sql([example], decomposer_model_path, completer_model_path)[0]
    print(sql)


