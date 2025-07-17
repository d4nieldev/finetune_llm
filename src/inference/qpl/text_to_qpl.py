import re
import json
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict
from enum import Enum
import logging as log
from pathlib import Path
import argparse

import torch
from peft import AutoPeftModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets import load_dataset
from tqdm import tqdm

from src.prompters.qpl import QPLDecomposerPrompter, QPLDecomposerCotPrompter, QPLCompleterPrompter
from src.utils.qpl.tree import QPLQDTree, Operator
from src.utils.generation import to_model_prompt, generate_batch
from src.utils.lists import flatten, unflatten


log.basicConfig(
    level=log.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class DecomposerExample(TypedDict):
    question: str
    db_id: str
    cot: str | None


class CompleterExample(TypedDict):
    question: str
    db_id: str
    prefix_qpl: str
    line_num: int
    op: str
    parent_question: Optional[str]


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
        if '-cot-' in decomposer_model_path:
            decomposer_prompter = QPLDecomposerCotPrompter(with_assistant=False)
        else:
            decomposer_prompter = QPLDecomposerPrompter(with_assistant=False)
        decomposer_model = AutoPeftModelForCausalLM.from_pretrained(decomposer_model_path, attn_implementation="eager").to("cuda")
        decomposer_tokenizer = AutoTokenizer.from_pretrained(decomposer_model_path)
        decomposer_model.eval()

        trees = decompose(
            examples=examples,
            prompter=decomposer_prompter,
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
    completer_prompter = QPLCompleterPrompter()
    completer_model = AutoPeftModelForCausalLM.from_pretrained(completer_model_path, attn_implementation="eager").to("cuda")
    completer_tokenizer = AutoTokenizer.from_pretrained(completer_model_path)
    completer_model.eval()
    complete(
        trees=valid_trees,
        prompter=completer_prompter,
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
    ) -> List[QPLQDTree]:
    with open(trees_ckpt_file, "r") as f:
        trees = json.load(f)
    
    trees = [QPLQDTree.from_dict(tree) for tree in trees]
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
        prompter: QPLDecomposerPrompter | QPLDecomposerCotPrompter,
        model: PreTrainedModel,
        mode: DecomposerMode,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        max_new_tokens: int,
        max_retries: int = 3,
    ) -> List[QPLQDTree]:

    def rec(examples: List[DecomposerExample], lvl: int = 1) -> List[QPLQDTree]:
        if len(examples) == 0:
            return []
        
        progress_bar = tqdm(total=len(examples), desc=f"Decomposing (level {lvl})", unit="question")

        # Create a QPL tree for each example
        trees = [QPLQDTree(question=ex['question'], db_id=ex['db_id']) for ex in examples]

        # Use the decomposer model to generate the questions for the next layer of each QPL tree
        # If llm is in sampling mode, retry the decomposition until a sub-question is different from its parent question
        chat_templates = list(map(prompter.to_chat_template, examples))
        prompts = list(map(lambda ct: to_model_prompt(tokenizer, ct), chat_templates))
        
        output_pattern = re.compile(r"(?P<reasoning><think>.*?</think>)?\s*(?P<answer>.*)", re.DOTALL)
        outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            model_prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_bar=progress_bar,
            is_valid_output=lambda i, output: (m := output_pattern.match(output)) is not None and examples[i]['question'] not in m.group("answer").split('\n'),
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
            if not (m := output_pattern.match(output)):
                log.warning(f"Output for question '{example['question']}' does not match the expected format. Got:\n{output}\nSkipping this tree.")
                children_examples.append([])
                continue
            lines = m.group('answer').split("\n")
            sub_questions = [l.strip() for l in lines[1:]]
            if example['question'] in sub_questions and mode == DecomposerMode.GREEDY:
                log.warning(f"Question '{example['question']}' is decomposed into itself. With greedy decoding, this can lead to an infinite loop - consider using sampling. Skipping this tree.")
                children_examples.append([])
                continue
            example['cot'] = cot if (cot := m.group('reasoning')) else None
            tree.op = Operator(lines[0].strip())
            children_examples.append([DecomposerExample(question=sub_question, db_id=example['db_id'], cot=None) for sub_question in sub_questions])

        # Generate the trees of the children questions
        flat_children_examples, lengths = flatten(children_examples)
        flat_children_trees = rec(flat_children_examples, lvl+1)
        children_trees = unflatten(flat_children_trees, lengths)

        # Assign the children trees to the parent trees
        for tree, children in zip(trees, children_trees):
            for child in children:
                child.parent = tree
            if len(children) == 0:
                tree.children = ()
            elif len(children) == 1:
                tree.children = (children[0],)
            else:
                tree.children = tuple(children)
        
        return trees
    
    return rec(examples)


def post_order_index_tree(tree: QPLQDTree, counter: int = 1) -> int:
    for child in tree.children:
        counter = post_order_index_tree(child, counter)
    tree.line_num = counter
    return counter + 1

@torch.no_grad()
def complete(
        trees: List[QPLQDTree],
        prompter: QPLCompleterPrompter,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        max_new_tokens: int,
    ) -> None:

    num_nodes = sum([tree.line_num for tree in trees])
    progress_bar = tqdm(total=num_nodes, desc="Completing QPL", unit="node")

    def rec(trees: List[QPLQDTree]) -> None:
        if len(trees) == 0:
            return
        
        # Complete the children before completing the parent
        trees_to_complete = []
        for tree in trees:
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
        chat_templates = list(map(prompter.to_chat_template, examples))
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
            if output is None:
                log.warning(f"Tree for question '{tree.question}' could not be completed. Skipping this tree.")
                continue
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
    examples = [DecomposerExample(question=ex['question'], db_id=ex['qpl'].split('|')[0].strip(), cot=None) for ex in nl2sql_dataset]

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
