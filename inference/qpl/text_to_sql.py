import json
from typing import Tuple, Union, Optional, List, Dict, Any
from typing_extensions import TypedDict
from dataclasses import dataclass
from enum import Enum

from inference.qpl.qpl_to_cte import flat_qpl_to_cte
from processors.qpl import QPLDecomposerProcessor, QPLComposerProcessor
from utils.generation_utils import to_model_prompt, generate_batch
from utils.lists import flatten, unflatten

import torch
from peft import AutoPeftModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets import load_dataset
from tqdm import tqdm


class DecomposerExample(TypedDict):
    question: str
    db_id: str


class CompleterExample(TypedDict):
    question: str
    db_id: str
    prefix_qpl: str
    op: str
    parent_question: Optional[str]


class Result(TypedDict):
    pred_cte: str
    pred_qpl: str


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
    op: Operator = None   # type: ignore
    qpl_line: str = None  # type: ignore
    line_idx: int = None  # type: ignore
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


def text_to_sql(
        examples: List[DecomposerExample], 
        decomposer_model_path: str, 
        completer_model_path: str,
        decomposer_bsz: int = 8,
        decomposer_max_new_tokens: int = 256,
        completer_bsz: int = 8,
        completer_max_new_tokens: int = 256,
    ) -> List[Result]:
    # Decompose input questions
    decomposer_processor = QPLDecomposerProcessor()
    decomposer_model = AutoPeftModelForCausalLM.from_pretrained(decomposer_model_path, attn_implementation="eager").to("cuda")
    decomposer_tokenizer = AutoTokenizer.from_pretrained(decomposer_model_path)
    decomposer_model.eval()
    trees = decompose(
        examples=examples,
        processor=decomposer_processor,
        model=decomposer_model,
        tokenizer=decomposer_tokenizer,
        batch_size=decomposer_bsz,
        max_new_tokens=decomposer_max_new_tokens,
    )
    decomposer_model = decomposer_model.to("cpu")
    
    # Post-order index the trees
    for tree in trees:
        post_order_index_tree(tree)
    
    torch.cuda.empty_cache()

    # complete QPL for trees
    completer_processor = QPLComposerProcessor()
    completer_model = AutoPeftModelForCausalLM.from_pretrained(completer_model_path, attn_implementation="eager").to("cuda")
    completer_tokenizer = AutoTokenizer.from_pretrained(completer_model_path)
    completer_model.eval()
    complete(
        trees=trees,
        processor=completer_processor,
        model=completer_model,
        tokenizer=completer_tokenizer,
        batch_size=completer_bsz,
        max_new_tokens=completer_max_new_tokens,
    )

    with open("output/qpl/trees.json", "w") as f:
        json.dump([tree.to_dict() for tree in trees], f, indent=2)

    # Convert QPL to SQL
    flat_qpls = [tree.qpl.split(' ; ') for tree in trees]
    db_ids = [tree.db_id for tree in trees]
    ctes = [flat_qpl_to_cte(flat_qpl, db_id) for flat_qpl, db_id in zip(flat_qpls, db_ids)]

    return [Result(pred_cte=cte, pred_qpl=tree.qpl) for tree, cte in zip(trees, ctes)]
    

@torch.no_grad()
def decompose(
        examples: List[DecomposerExample],
        processor: QPLDecomposerProcessor,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 8,
        max_new_tokens: int = 256,
    ) -> List[QPLTree]:

    def rec(examples: List[DecomposerExample], lvl: int = 1) -> List[QPLTree]:
        if len(examples) == 0:
            return []
        
        progress_bar = tqdm(total=len(examples), desc=f"Decomposing (level {lvl})", unit="question")

        # Create a QPL tree for each example
        trees = [QPLTree(question=ex['question'], db_id=ex['db_id']) for ex in examples]

        # Use the decomposer model to generate the questions for the next layer of each QPL tree
        chat_templates = list(map(processor.to_chat_template, examples))
        prompts = list(map(lambda ct: to_model_prompt(tokenizer, ct), chat_templates))
        outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            model_prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_bar=progress_bar,
        )

        # Parse the outputs and create child QPL trees
        children_examples = []
        for example, tree, output in zip(examples, trees, outputs):
            lines = output.split("\n")
            tree.op = Operator(lines[0].strip())
            children_examples.append([DecomposerExample(question=l.strip(), db_id=example['db_id']) for l in lines[1:]])

        # Generate the trees of the children questions
        flat_children_examples, lengths = flatten(children_examples)
        flat_children_trees = rec(flat_children_examples, lvl+1)
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

    return rec(examples)


def post_order_index_tree(tree: QPLTree, counter: int = 1) -> int:
    if tree.children:
        for child in tree.children:
            counter = post_order_index_tree(child, counter)
    tree.line_idx = counter
    return counter + 1

@torch.no_grad()
def complete(
        trees: List[QPLTree],
        processor: QPLComposerProcessor,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 16,
        max_new_tokens: int = 128,
    ) -> None:

    num_nodes = sum([tree.line_idx for tree in trees])
    progress_bar = tqdm(total=num_nodes, desc="Completing QPL", unit="node")

    def rec(trees: List[QPLTree]) -> None:
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

        rec(trees_to_complete)
        
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
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_bar=progress_bar,
        )

        # Parse the outputs and assign the QPL lines to the trees
        for tree, output in zip(trees, outputs):
            completion = output.split('\n')[0].strip()
            tree.qpl_line = f"#{tree.line_idx} = {tree.op.value} {completion}"

    return rec(trees)


if __name__ == "__main__":
    decomposer_model_path = "output/gemma-3-4b-it-question_decomposer_ds_train_batch_size=1_gradient_accumulation_steps=1_learning_rate=0.0002_num_train_epochs=2_gradient_checkpointing=False_logging_steps=500_save_steps=5000_random_seed=1_lora=True_r=16_alpha=32_dropout=0.05/checkpoint-20958"
    completer_model_path = "output/erbz0056_gemma-3-4b-it-qpl_composer_train_batch_size=1_gradient_accumulation_steps=8_learning_rate=0.0002_num_train_epochs=4_gradient_checkpointing=True_logging_steps=0.05_save_steps=0.5_random_seed=1_lora=True_r=16_alpha=32_dropout=0.05/checkpoint-5316"
    decomposer_bsz = 8
    decomposer_max_new_tokens = 256
    completer_bsz = 8
    completer_max_new_tokens = 256

    nl2sql_dataset = list(load_dataset("d4nieldev/nl2qpl-ds", split="validation"))
    examples = [DecomposerExample(question=ex['question'], db_id=ex['qpl'].split('|')[0].strip()) for ex in nl2sql_dataset]

    results = text_to_sql(
        examples=examples,
        decomposer_model_path=decomposer_model_path,
        completer_model_path=completer_model_path,
        decomposer_bsz=decomposer_bsz,
        decomposer_max_new_tokens=decomposer_max_new_tokens,
        completer_bsz=completer_bsz,
        completer_max_new_tokens=completer_max_new_tokens,
    )
    
    enriched_results = [dict(result) for result in results]
    for result, example in zip(enriched_results, nl2sql_dataset):
        result['gold_qpl'] = example['qpl'].split('|')[1].strip()
        result['gold_cte'] = example['cte']
        result['gold_sql'] = example['query']

    with open("output/qpl/results.json", "w") as f:
        json.dump(enriched_results, f, indent=2)
