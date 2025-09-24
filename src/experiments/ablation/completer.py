import argparse
import json

import torch
from unsloth import FastLanguageModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoModelForCausalLM
from datasets import load_dataset

from src.prompters import QPLCompleterCotPrompter
from src.databuilders.completer.build import get_decomposer_roots
from src.utils.tree import PartialQDTree, QPLQDTree
from src.evaluation.text_to_qpl import complete
import src.utils.paths as p
from src.utils.paths import TRAINED_MODELS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="QPL Completer Ablation")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for processing")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of new tokens to generate")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load model & tokenizer
    # model = AutoModelForCausalLM.from_pretrained(TRAINED_MODELS_DIR / args.model_dir, attn_implementation='flash_attention_2', torch_dtype=torch.float16).to('cuda')
    # tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODELS_DIR / args.model_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(TRAINED_MODELS_DIR / args.model_dir),
        max_seq_length = 16*1024,
        load_in_4bit = True,
        load_in_8bit = False,
        # fast_inference = True, # uses vLLM
    )
    model = model.eval()

    # Load and process data
    def partial_qd_to_qd(tree: PartialQDTree) -> QPLQDTree:
        """Convert a PartialQDTree to a QPLQDTree."""
        qd_tree = QPLQDTree(
            question=tree.question,
            db_id=tree.db_id,
            op=tree.op,
        )
        if tree.children:
            qd_tree.children = tuple(partial_qd_to_qd(child) for child in tree.children)
            for child in qd_tree.children:
                child.parent = qd_tree
        return qd_tree


    prompter = QPLCompleterCotPrompter(with_assistant=False, schema_representation="m_schema")
    decomposer_data = load_dataset("bgunlp/question_decomposer_ds", split="validation")
    nl2qpl_data = load_dataset('d4nieldev/nl2qpl-ds', split='validation')
    root_questions = set(row['question'] for row in nl2qpl_data)
    decomposer_data = [row for row in decomposer_data if row['question'] not in [row['sub_question_1'], row['sub_question_2']]]
    root_qd_trees = get_decomposer_roots(decomposer_data, root_questions)
    root_qd_trees = [partial_qd_to_qd(tree) for tree in root_qd_trees]

    def post_order_index_tree(tree: QPLQDTree, counter: int = 1) -> int:
        for child in tree.children:
            counter = post_order_index_tree(child, counter)
        tree.line_num = counter
        return counter + 1

    for tree in root_qd_trees:
        post_order_index_tree(tree)

    # Complete QPL for each tree
    complete(
        trees=root_qd_trees,
        prompter=prompter,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    output_path = p.ABLATION_COMPLETER_OUTPUT_DIR / args.model_dir / "full_tree_outputs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([tree.to_dict() for tree in root_qd_trees], f, indent=2)