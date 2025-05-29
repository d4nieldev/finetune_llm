import json
from typing import Tuple, Union, Optional, List, Dict, Any
from typing_extensions import TypedDict
from dataclasses import dataclass
from enum import Enum
import logging as log
from pathlib import Path
import argparse

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

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DecomposerExample(TypedDict):
    question: str
    db_id: str


class CompleterExample(TypedDict):
    question: str
    db_id: str
    prefix_qpl: str
    line_num: int
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
    op: Operator = None   # type: ignore
    line_num: int = None  # type: ignore
    qpl_line: str = None  # type: ignore
    parent: Optional["QPLTree"] = None
    children: Optional[Union[Tuple["QPLTree"], Tuple["QPLTree", "QPLTree"]]] = None

    def to_dict(self) -> Dict[str, Any]:        
        output = {
            "db_id": self.db_id,
            "question": self.question,
            "is_valid": self.is_valid,
            "line_num": self.line_num,
        }

        if self.is_valid:
            output = {
                **output,
                "op": self.op.value,
                "qpl": self.qpl,
                "prefix_qpl": self.prefix_qpl,
                "qpl_line": self.qpl_line,
                "children": [child.to_dict() for child in self.children] if self.children else None,
            }

        return output
    

    @staticmethod
    def from_dict(tree_dict: Dict[str, Any]) -> "QPLTree":
        tree = QPLTree(
            question=tree_dict["question"],
            db_id=tree_dict["db_id"],
            op=Operator(tree_dict["op"]) if tree_dict['is_valid'] else None,
            line_num=tree_dict["line_num"],
            qpl_line=tree_dict.get("qpl_line"),
        )
        if tree_dict.get("children"):
            tree.children = tuple(QPLTree.from_dict(child) for child in tree_dict["children"])
            for child in tree.children:
                child.parent = tree
        return tree

    @property
    def prefix_qpl(self) -> Optional[str]:
        try:
            if self.children is None:
                return ""
            return "\n".join([(child.prefix_qpl + "\n" + child.qpl_line).strip() for child in self.children]).replace("\n", " ; ")
        except Exception as e:
            return None
    
    @property
    def qpl(self) -> Optional[str]:
        try:
            return f"{self.prefix_qpl} ; {self.qpl_line}" if self.prefix_qpl else self.qpl_line
        except Exception as e:
            return None
    
    @property
    def is_valid(self) -> bool:
        return self.op and (all(child.is_valid for child in self.children) if self.children else True)
    

class DecomposerMode(Enum):
    GREEDY = "greedy"
    SAMPLING = "sampling"
    FIRST_GREEDY = "first_greedy"


class Result(TypedDict):
    db_id: str
    question: str
    pred_qpl: Optional[str]


def text_to_qpl(
        examples: List[DecomposerExample], 
        decomposer_model_path: str, 
        completer_model_path: str,
        decomposer_mode: DecomposerMode,
        decomposer_trees_path: Path,
        completed_trees_path: Path,
        decomposer_bsz: int = 8,
        decomposer_max_new_tokens: int = 256,
        completer_bsz: int = 8,
        completer_max_new_tokens: int = 256,
        is_load_decomposer_trees: bool = False,
    ) -> List[Result]:
    # Decompose input questions
    if not is_load_decomposer_trees:
        decomposer_processor = QPLDecomposerProcessor(with_assistant=False)
        decomposer_model = AutoPeftModelForCausalLM.from_pretrained(decomposer_model_path, attn_implementation="eager").to("cuda")
        decomposer_tokenizer = AutoTokenizer.from_pretrained(decomposer_model_path)
        decomposer_model.eval()

        trees = decompose(
            examples=examples,
            processor=decomposer_processor,
            model=decomposer_model,
            mode=decomposer_mode,
            tokenizer=decomposer_tokenizer,
            batch_size=decomposer_bsz,
            max_new_tokens=decomposer_max_new_tokens,
        )
        decomposer_model = decomposer_model.to("cpu")
        
        # Post-order index the trees
        for tree in trees:
            post_order_index_tree(tree)

        # Save checkpoint (natural language query decomposition)    
        decomposer_trees_path.write_text(json.dumps([tree.to_dict() for tree in trees], indent=2))
        log.info(f"Saved natural language queries decomposition trees (no code) to '{decomposer_trees_path}'")
    else:
        trees = load_decomposer_trees(decomposer_trees_path)

    valid_trees = [tree for tree in trees if tree.is_valid]
    log.info(f"Filtered out {len(trees) - len(valid_trees)} invalid trees from {len(trees)} trees")

    # complete QPL for trees
    torch.cuda.empty_cache()
    completer_processor = QPLComposerProcessor()
    completer_model = AutoPeftModelForCausalLM.from_pretrained(completer_model_path, attn_implementation="eager").to("cuda")
    completer_tokenizer = AutoTokenizer.from_pretrained(completer_model_path)
    completer_model.eval()
    complete(
        trees=valid_trees,
        processor=completer_processor,
        model=completer_model,
        tokenizer=completer_tokenizer,
        batch_size=completer_bsz,
        max_new_tokens=completer_max_new_tokens,
    )

    # Save checkpoint (QPL generation for each node)
    completed_trees_path.write_text(json.dumps([tree.to_dict() for tree in trees], indent=2))
    log.info(f"Saved QPL generation trees to  '{completed_trees_path}'")

    return [Result(db_id=tree.db_id, question=tree.question, pred_qpl=tree.qpl) for tree in trees]


def load_decomposer_trees(
        trees_ckpt_file: Path,
    ) -> List[QPLTree]:
    with open(trees_ckpt_file, "r") as f:
        trees = json.load(f)
    
    trees = [QPLTree.from_dict(tree) for tree in trees]
    log.info(f"Loaded natural language queries decomposition trees (no code) from '{trees_ckpt_file}'")

    return trees


def get_decomposer_generation_params(mode: DecomposerMode) -> Dict[str, Any]:
    if mode == DecomposerMode.GREEDY:
        return {
            'do_sample': False,
            'top_p': None,
            'top_k': None,
        }
    elif mode == DecomposerMode.SAMPLING:
        return {
            'do_sample': True,
            'top_p': 0.95,
            'top_k': 50,
        }
    elif mode == DecomposerMode.FIRST_GREEDY:
        return {
            'do_sample': True,
            'top_p': 0.95,
            'top_k': 50,
            'first_greedy': True,
        }
    raise ValueError(f"Unknown decomposer mode: {mode}")
    

@torch.no_grad()
def decompose(
        examples: List[DecomposerExample],
        processor: QPLDecomposerProcessor,
        model: PreTrainedModel,
        mode: DecomposerMode,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        max_new_tokens: int,
        max_retries: int = 3,
    ) -> List[QPLTree]:

    def rec(examples: List[DecomposerExample], lvl: int = 1) -> List[QPLTree]:
        if len(examples) == 0:
            return []
        
        progress_bar = tqdm(total=len(examples), desc=f"Decomposing (level {lvl})", unit="question")

        # Create a QPL tree for each example
        trees = [QPLTree(question=ex['question'], db_id=ex['db_id']) for ex in examples]

        # Use the decomposer model to generate the questions for the next layer of each QPL tree
        # If llm is in sampling mode, retry the decomposition until a sub-question is different from its parent question
        chat_templates = list(map(processor.to_chat_template, examples))
        prompts = list(map(lambda ct: to_model_prompt(tokenizer, ct), chat_templates))
        
        outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            model_prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_bar=progress_bar,
            is_valid_output=(lambda i, output: examples[i]['question'] != output) if mode in [DecomposerMode.FIRST_GREEDY, DecomposerMode.SAMPLING] else None,
            max_retries=max_retries,
            **get_decomposer_generation_params(mode)
        )
        progress_bar.close()

        # Parse the outputs and create child QPL trees
        children_examples = []
        for example, tree, output in zip(examples, trees, outputs):
            if output is None:
                log.warning(f"Question '{example['question']}' is decomposed into itself after {max_retries} tries. Skipping this tree.")
                children_examples.append([])
                continue
            lines = output.split("\n")
            sub_questions = [l.strip() for l in lines[1:]]
            if example['question'] in sub_questions and mode == DecomposerMode.GREEDY:
                log.warning(f"Question '{example['question']}' is decomposed into itself. With greedy decoding, this can lead to an infinite loop - consider using sampling. Skipping this tree.")
                children_examples.append([])
                continue
            tree.op = Operator(lines[0].strip())
            children_examples.append([DecomposerExample(question=sub_question, db_id=example['db_id']) for sub_question in sub_questions])

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
    tree.line_num = counter
    return counter + 1

@torch.no_grad()
def complete(
        trees: List[QPLTree],
        processor: QPLComposerProcessor,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        max_new_tokens: int,
    ) -> None:

    num_nodes = sum([tree.line_num for tree in trees])
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
                line_num=tree.line_num,
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
            do_sample=False,  # QPL generation should be deterministic
        )

        # Parse the outputs and assign the QPL lines to the trees
        for tree, output in zip(trees, outputs):
            completion = output.split('\n')[0].strip()
            tree.qpl_line = f"#{tree.line_num} = {tree.op.value} {completion}"

    return rec(trees)


def parse_args():
    parser = argparse.ArgumentParser(description="Text to SQL")

    # Mandatory
    parser.add_argument("--decomposer_model_path", type=str, required=False, help="Path to the decomposer model")
    parser.add_argument("--load_decomposer_trees", action="store_true", help="Load the decomposer trees from the specified path")
    parser.add_argument("--completer_model_path", type=str, required=True, help="Path to the completer model")
    parser.add_argument("--decomposer_trees_path", type=Path, required=True, help="Where to save the trees generated by the decomposer")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the results")
    
    # Optional
    parser.add_argument("--decomposer_mode", type=DecomposerMode, choices=list(DecomposerMode), default=DecomposerMode.FIRST_GREEDY, help="Decomposer model mode")
    parser.add_argument("--completed_trees_path", type=Path, required=False, help="Where to save the completed trees (QPL generation). Defaults to decomposer_trees_path with 'completed_' prefix")
    parser.add_argument("--decomposer_bsz", type=int, default=8, help="Batch size for the decomposer model")
    parser.add_argument("--decomposer_max_new_tokens", type=int, default=256, help="Max new tokens for the decomposer model")
    parser.add_argument("--completer_bsz", type=int, default=4, help="Batch size for the completer model")
    parser.add_argument("--completer_max_new_tokens", type=int, default=256, help="Max new tokens for the completer model")

    args = parser.parse_args()

    if not (bool(args.decomposer_model_path) ^ args.load_decomposer_trees):
        parser.error("Either --decomposer_model_path or --load_decomposer_trees must be provided, but not both.")

    if not args.completed_trees_path:
        args.completed_trees_path = args.decomposer_trees_path.with_name(f"completed_{args.decomposer_trees_path.name}")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    nl2sql_dataset = list(load_dataset("d4nieldev/nl2qpl-ds", split="validation"))
    examples = [DecomposerExample(question=ex['question'], db_id=ex['qpl'].split('|')[0].strip()) for ex in nl2sql_dataset]

    results = text_to_qpl(
        examples=examples,
        decomposer_model_path=args.decomposer_model_path,
        completer_model_path=args.completer_model_path,
        decomposer_mode=args.decomposer_mode,
        decomposer_trees_path=args.decomposer_trees_path,
        completed_trees_path=args.completed_trees_path,
        decomposer_bsz=args.decomposer_bsz,
        decomposer_max_new_tokens=args.decomposer_max_new_tokens,
        completer_bsz=args.completer_bsz,
        completer_max_new_tokens=args.completer_max_new_tokens,
        is_load_decomposer_trees=args.load_decomposer_trees,
    )
    
    assert len(results) == len(nl2sql_dataset), f"Expected {len(nl2sql_dataset)} results, but got {len(results)}"

    enriched_results = [dict(result) for result in results]
    for result, example in zip(enriched_results, nl2sql_dataset):
        db_id = example['qpl'].split('|')[0].strip()
        assert db_id == result['db_id'], f"DB ID mismatch: {db_id} != {result['db_id']}"
        result['gold_qpl'] = example['qpl'].split('|')[1].strip()
        result['gold_cte'] = example['cte']
        result['gold_sql'] = example['query']

    with open(args.output_file, "w") as f:
        json.dump(enriched_results, f, indent=2)
