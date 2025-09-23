import asyncio
from pathlib import Path

import litellm
from litellm import acompletion
from litellm.caching.caching import Cache, LiteLLMCacheType
from aiolimiter import AsyncLimiter
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

from src.utils.chat_types import Message
from src.utils.paths import DB_SCHEMAS_JSON_PATH
from src.utils.qpl.schema import DBSchema

litellm.cache = Cache(type=LiteLLMCacheType.DISK)


prompt_dir = Path(__file__).parent / "prompt"
system_prompt = (prompt_dir / "system.md").read_text()
user_prompt_template = (prompt_dir / "user.md").read_text()


async def generate_CoT(
        model: str, 
        schemas: dict[str, DBSchema], 
        example: dict,
        limiter: AsyncLimiter) -> dict:
    
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
    if example['db_id'] == 'car_11':
        db_schema = schemas['car_1']
    else:
        db_schema = schemas[example['db_id']]
    user_prompt = user_prompt_template.format(
        db_id=example['db_id'],
        db_schema=db_schema,
        question=example['question'],
        op=example['op'],
        sub_questions=sub_questions_str,
        qpl=(example['prefix_qpl'] + "\n" + example['qpl_line'] + " ;").strip(),
    )
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    async with limiter:
        resp = await acompletion(model=model, messages=messages, caching=True)
    cot = resp.choices[0]['message']['content']
    return {
        "db_id": example['db_id'],
        "question": example['question'],
        "op": example['op'],
        "sub_question_1": example['sub_question_1'],
        "sub_question_2": example['sub_question_2'],
        "cot": cot,
    }


async def get_examples(model: str, split: str, model_rpm: int = 50) -> list[dict]:
    schemas = DBSchema.from_db_schemas_file(DB_SCHEMAS_JSON_PATH, apply_lower=False)
    decomposer_completer_ds = load_dataset("d4nieldev/qpl-decomposer-completer-ds", split=split)
    pbar = tqdm(total=len(decomposer_completer_ds), desc=f"Generating {split!r} CoTs")
    limiter = AsyncLimiter(max_rate=model_rpm, time_period=60)
    pbar_lock = asyncio.Lock()
    
    async def update_tqdm(task):
        result = await task
        async with pbar_lock:
            pbar.update(1)
        return result
    
    tasks = [
        update_tqdm(generate_CoT(model=model, schemas=schemas, example=example, limiter=limiter))
        for example in decomposer_completer_ds
    ]

    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    MODEL = 'anthropic/claude-4-sonnet-20250514'
    schemas = DBSchema.from_db_schemas_file(DB_SCHEMAS_JSON_PATH, apply_lower=False)

    dataset = {}
    for split in ['train', 'validation']:
        dataset[split] = asyncio.run(get_examples(model=MODEL, split=split, model_rpm=50))
    
    ds = DatasetDict({
        split: Dataset.from_list(data)
        for split, data in dataset.items()
    })
    ds.push_to_hub("d4nieldev/qpl-decomposer-cot-ds", "original")
