import time
import json
import random
import asyncio
from pathlib import Path

from litellm import acompletion
from tqdm import tqdm

from src.utils.chat_types import ChatMessage
from src.utils.qpl.paths import DB_SCHEMAS_JSON_PATH
from src.utils.qpl.schema import DBSchema
from src.utils.qpl.tree import PartialQDTree, Operator
from databuilders.completer.create_completer_data import load_qd_trees


system_prompt = Path("src/inference/qpl/decomposer_cot/prompt/system.md").read_text()
user_prompt_template = Path("src/inference/qpl/decomposer_cot/prompt/user.md").read_text()

async def generate_CoT(model: str, schemas: dict[str, DBSchema], example: dict, pbar: tqdm | None = None) -> dict:
    if not example['sub_question_1'] and not example['sub_question_2']:
        sub_questions_str = ""
    elif example['sub_question_1'] and not example['sub_question_2']:
        sub_questions_str = f"Sub-Question: {example['sub_question_1']}"
    elif not example['sub_question_1'] and example['sub_question_2']:
        sub_questions_str = f"Sub-Question: {example['sub_question_2']}"
    else:
        sub_questions_str = (
            f"Sub-Question 1: {example['sub_question_1']}\n"
            f"Sub-Question 2: {example['sub_question_2']}"
        )
    user_prompt = user_prompt_template.format(
        db_id=example['db_id'],
        db_schema=schemas[example['db_id']],
        question=example['question'],
        op=example['op'],
        sub_questions=sub_questions_str,
        qpl=example['qpl'],
    )
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]
    resp = await acompletion(model=model, messages=messages)
    example['cot'] = resp.choices[0]['message']['content']
    if pbar:
        async with asyncio.Lock():
            pbar.update(1)
    return example


def get_examples(split: str = "validation") -> list[dict]:
    qd_trees = load_qd_trees(split=split)

    def tree_nodes(qd_tree: PartialQDTree) -> list[dict]:
        if not qd_tree.op or not qd_tree.qpl_line:
            return []
        
        qpl = "\n".join([qd_tree.prefix_qpl or "", qd_tree.qpl_line]).strip()
        
        nodes = [
            {
                "db_id": qd_tree.db_id,
                "question": qd_tree.question,
                "op": qd_tree.op.value,
                "sub_question_1": qd_tree.children[0].question if len(qd_tree.children) > 0 else None,
                "sub_question_2": qd_tree.children[1].question if len(qd_tree.children) > 1 else None,
                "qpl": qpl,
                "num_lines": len(qpl.split("\n")),
            }
        ]

        if len(qd_tree.children) > 0:
            for child in qd_tree.children:
                nodes += tree_nodes(child)

        return nodes

    return [ex for qd_tree in qd_trees for ex in tree_nodes(qd_tree)]

async def main(model: str):
    schemas = DBSchema.from_db_schemas_file(DB_SCHEMAS_JSON_PATH, apply_lower=False)

    examples = get_examples(split="validation")[:3]
    
    pbar = tqdm(total=len(examples), desc="Generating CoTs")

    tasks = [
        generate_CoT(model=model, schemas=schemas, example=example, pbar=pbar)
        for example in examples
    ]

    results = await asyncio.gather(*tasks)

    Path("output/qpl/decomposer_cot/results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    random.seed(1)
    asyncio.run(main(model='anthropic/claude-4-sonnet-20250514'))
