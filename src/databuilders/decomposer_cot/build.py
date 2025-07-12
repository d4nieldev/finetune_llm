import asyncio
from pathlib import Path

from litellm import acompletion
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

from src.utils.chat_types import ChatMessage
from src.utils.qpl.paths import DB_SCHEMAS_JSON_PATH
from src.utils.qpl.schema import DBSchema


prompt_dir = Path(__file__).parent / "prompt"
system_prompt = (prompt_dir / "system.md").read_text()
user_prompt_template = (prompt_dir / "user.md").read_text()

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
        qpl=(example['prefix_qpl'] + "\n" + example['qpl_line'] + " ;").strip(),
    )
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]
    resp = await acompletion(model=model, messages=messages)
    cot = resp.choices[0]['message']['content']
    if pbar:
        async with asyncio.Lock():
            pbar.update(1)
    return {
        "db_id": example['db_id'],
        "question": example['question'],
        "cot": cot,
        "op": example['op'],
        "sub_question_1": example['sub_question_1'],
        "sub_question_2": example['sub_question_2'],
    }


async def get_examples(model: str, split: str) -> list[dict]:
    schemas = DBSchema.from_db_schemas_file(DB_SCHEMAS_JSON_PATH, apply_lower=False)
    decomposer_completer_ds = load_dataset("d4nieldev/qpl-decomposer-completer-ds", split=split)
    pbar = tqdm(total=len(decomposer_completer_ds), desc=f"Generating {split!r} CoTs")
    tasks = [
        generate_CoT(model=model, schemas=schemas, example=example, pbar=pbar)
        for example in decomposer_completer_ds
    ]

    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    MODEL = 'anthropic/claude-4-sonnet-20250514'
    schemas = DBSchema.from_db_schemas_file(DB_SCHEMAS_JSON_PATH, apply_lower=False)

    dataset = {}
    for split in ['train', 'validation']:
        dataset[split] = asyncio.run(get_examples(model=MODEL, split=split))
    
    ds = DatasetDict({
        split: Dataset.from_list(data)
        for split, data in dataset.items()
    })
    ds.push_to_hub("d4nieldev/qpl-decomposer-cot-ds")
