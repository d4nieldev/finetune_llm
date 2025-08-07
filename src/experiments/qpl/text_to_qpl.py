import re
import json
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict
from enum import Enum
import logging as log
from pathlib import Path
import argparse

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets import load_dataset
from tqdm import tqdm

from src.prompters.qpl import QPLDecomposerPrompter, QPLDecomposerCotPrompter, QPLCompleterPrompter, QPLCompleterCotPrompter
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


class CompleterExample(TypedDict):
    question: str
    db_id: str
    prefix_qpl: str
    line_num: int
    op: str
    children_str: str
    parent_question: Optional[str]


class GenerationMode(Enum):
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
        mode: GenerationMode,
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
        if '-cot' in decomposer_model_path:
            log.info("Using COT decomposer prompter")
            decomposer_prompter = QPLDecomposerCotPrompter(with_assistant=False)
        else:
            log.info("Using standard decomposer prompter")
            decomposer_prompter = QPLDecomposerPrompter(with_assistant=False)
        decomposer_model = AutoModelForCausalLM.from_pretrained(decomposer_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16).to("cuda")
        decomposer_tokenizer = AutoTokenizer.from_pretrained(decomposer_model_path)
        decomposer_model.eval()

        trees = decompose(
            examples=examples,
            prompter=decomposer_prompter,
            model=decomposer_model,
            mode=mode,
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
    if '-cot' in completer_model_path:
        log.info("Using COT completer prompter")
        completer_prompter = QPLCompleterCotPrompter(with_assistant=False)
    else:
        log.info("Using standard completer prompter")
        completer_prompter = QPLCompleterPrompter(with_assistant=False)
    completer_model = AutoModelForCausalLM.from_pretrained(completer_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16).to("cuda")
    completer_tokenizer = AutoTokenizer.from_pretrained(completer_model_path)
    completer_model.eval()
    complete(
        trees=valid_trees,
        prompter=completer_prompter,
        model=completer_model,
        tokenizer=completer_tokenizer,
        batch_size=completer_bsz,
        max_new_tokens=completer_max_new_tokens,
        mode=mode,
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


def get_generation_params(mode: GenerationMode) -> Dict[str, Any]:
    if mode == GenerationMode.GREEDY:
        return {
            'do_sample': False,
            'top_p': None,
            'top_k': None,
        }
    elif mode == GenerationMode.SAMPLING:
        return {
            'do_sample': True,
            'top_p': 0.95,
            'top_k': 50,
            'temperature': 0.6,
        }
    elif mode == GenerationMode.FIRST_GREEDY:
        return {
            'do_sample': True,
            'top_p': 0.95,
            'top_k': 50,
            'temperature': 0.6,
            'first_greedy': True,
        }
    raise ValueError(f"Unknown decomposer mode: {mode}")
    

@torch.no_grad()
def decompose(
        examples: List[DecomposerExample],
        prompter: QPLDecomposerPrompter | QPLDecomposerCotPrompter,
        model: PreTrainedModel,
        mode: GenerationMode,
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
        def validate_decomposer_output(i: int, output: str) -> bool:
            m = output_pattern.match(output)
            if not m:
                log.warning(f"Output for question '{examples[i]['question']}' does not match the expected format. Got:\n{output}.")
                return False
            elif examples[i]['question'] in m.group("answer").split("\n"):
                log.warning(f"Question '{examples[i]['question']}' is decomposed into itself. Got:\n{output}.")
                return False
            else:
                try:
                    Operator(m.group('answer').split("\n")[0].strip())
                except ValueError:
                    log.warning(f"Output for question '{examples[i]['question']}' does not start with a valid operator. Got:\n{output}.")
                    return False
            return True
        
        outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            model_prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_bar=progress_bar,
            is_valid_output=validate_decomposer_output,
            max_retries=max_retries,
            **get_generation_params(mode)
        )
        progress_bar.close()

        # Parse the outputs and create child QPL trees
        children_examples = []
        for example, tree, output in zip(examples, trees, outputs):
            if output is None:
                log.error(f"Could not decompose question '{example['question']}' after {max_retries} tries. Stopping exploration.")
                children_examples.append([])
                continue
            if not (m := output_pattern.match(output)):
                raise ValueError("This should not happen, the output should always match the expected format.")
            lines = [l.strip() for l in m.group('answer').split("\n") if l.strip()]
            tree.decomposition_cot = cot if (cot := m.group('reasoning')) else None
            tree.op = Operator(lines[0].strip())
            sub_questions = [l.strip() for l in lines[1:]]
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
        max_retries: int = 3,
        mode: GenerationMode = GenerationMode.SAMPLING,
    ) -> None:

    num_nodes = sum([tree.line_num for tree in trees])
    progress_bar = tqdm(total=num_nodes, desc="Completing QPL", unit="node")

    def get_line_prefix(tree: QPLQDTree) -> str:
        children_str="Table" if tree.op == Operator.SCAN else f"[ {', '.join([f'#{child.line_num}' for child in tree.children])} ]"
        return f"#{tree.line_num} = {tree.op.value} {children_str}"

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
                op=str(tree.op),
                children_str="Table" if tree.op == Operator.SCAN else f"[ {', '.join([f'#{child.line_num}' for child in tree.children])} ]",
                parent_question=tree.parent.question if tree.parent else None,
            )
            for tree in trees
        ]
        chat_templates = list(map(prompter.to_chat_template, examples))
        prompts = list(map(lambda ct: to_model_prompt(tokenizer, ct), chat_templates))

        output_pattern = re.compile(r"(?P<reasoning><think>.*?</think>)?\s*```QPL\n(?P<answer>.*)\n```", re.DOTALL)
        def validate_completer_output(i: int, output: str) -> bool:
            # line_prefix = get_line_prefix(trees[i])
            if not (m := output_pattern.search(output)):
                log.warning(f"Output for question '{examples[i]['question']}' does not match the expected format. Got:\n{output}.")
                return False
            # this is too strict because of spaces
            # elif not m.group('answer').strip().startswith(line_prefix):
            #     log.warning(f"Output for question '{examples[i]['question']}' does not start with the expected line prefix ({line_prefix}). Got:\n{output}.")
            #     return False
            return True
    
        outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            model_prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progress_bar=progress_bar,
            is_valid_output=validate_completer_output,
            max_retries=max_retries,
            **get_generation_params(mode),
        )

        # Parse the outputs and assign the QPL lines to the trees
        for tree, output in zip(trees, outputs):
            if output is None:
                log.error(f"Tree for question '{tree.question}' could not be completed. Skipping this tree.")
                continue
            if not (m := output_pattern.search(output)):
                raise ValueError("This should not happen, the output should always match the expected format.")
            tree.completion_cot = m.group('reasoning')
            tree.qpl_line = m.group('answer').strip()

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
    parser.add_argument("--generation_mode", type=GenerationMode, choices=list(GenerationMode), default=GenerationMode.SAMPLING, help="Models decoding strategy")
    parser.add_argument("--completed_trees_path", type=Path, required=False, help="Where to save the completed trees (QPL generation). Defaults to decomposer_trees_path with 'completed_' prefix")
    parser.add_argument("--decomposer_bsz", type=int, default=4, help="Batch size for the decomposer model")
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
        mode=args.generation_mode,
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
