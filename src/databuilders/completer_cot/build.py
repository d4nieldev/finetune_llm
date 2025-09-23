import re
import json
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
        limiter: AsyncLimiter | None
    ) -> dict:
    
    if example['db_id'] == 'car_11':
        db_schema = schemas['car_1']
    else:
        db_schema = schemas[example['db_id']]

    flat_qpl_scan_pattern = re.compile(
        r"#(?P<idx>\d+) = Scan Table \[ (?P<table>\w+) \]( Predicate \[ (?P<pred>[^\]]+) \])?( Distinct \[ (?P<distinct>true) \])? Output \[ (?P<out>[^\]]+) \]"
    )
    flat_qpl_line_pattern = re.compile(
        r"#(?P<idx>\d+) = (?P<op>\w+) \[ (?P<ins>[^\]]+) \] ((?P<opt>\w+) \[ (?P<arg>[^\]]+) \] )*Output \[ (?P<out>[^\]]+) \]"
    )

    highlighted_qpl_line = example['qpl_line']
    if (m := flat_qpl_scan_pattern.match(example['qpl_line'])):
        given = f"#{m.group('idx')} = Scan Table"
        highlighted_qpl_line = f"*{given}* {highlighted_qpl_line.replace(given, '')}"
    elif (m := flat_qpl_line_pattern.match(example['qpl_line'])):
        given = f"#{m.group('idx')} = {m.group('op')} [ {m.group('ins')} ]"
        highlighted_qpl_line = f"*{given}* {highlighted_qpl_line.replace(given, '')}"
    else:
        raise ValueError(f"QPL line does not match expected patterns: {example['qpl_line']}")

    user_prompt = user_prompt_template.format(
        db_id=example['db_id'],
        db_schema=db_schema,
        question=example['question'],
        prefix_qpl=example['prefix_qpl'],
        highlighted_qpl_line=highlighted_qpl_line,
    )
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    if limiter:
        async with limiter:
            resp = await acompletion(model=model, messages=messages, caching=True)
    else:  
        resp = await acompletion(model=model, messages=messages, caching=True)
    cot = resp.choices[0]['message']['content']
    return {
        "db_id": example['db_id'],
        "parent_question": example['parent_question'],
        "question": example['question'],
        "prefix_qpl": example['prefix_qpl'],
        "op": example['op'],
        "cot": cot,
        "qpl_line": example['qpl_line'],
    }


def group_by_sample(ds: Dataset, col: str) -> Dataset:
    """Return a new Dataset that keeps the FIRST row for each unique value in `col`."""
    seen = set()
    keep_indices = []
    shuffled_ds = ds.shuffle(seed=42)
    for idx, value in enumerate(shuffled_ds[col]):
        if value not in seen:          # haven’t seen this value yet → keep this row
            seen.add(value)
            keep_indices.append(idx)
    return shuffled_ds.select(keep_indices)


async def get_examples(model: str, split: str, model_rpm: int = 50, no_limiter_i: int | None = None) -> list[dict]:
    schemas = DBSchema.from_db_schemas_file(DB_SCHEMAS_JSON_PATH, apply_lower=False)
    decomposer_completer_ds = load_dataset("d4nieldev/qpl-decomposer-completer-ds", split=split)
    # decomposer_completer_ds = group_by_sample(decomposer_completer_ds, "op")
    pbar = tqdm(total=len(decomposer_completer_ds), desc=f"Generating {split!r} CoTs")
    limiter = AsyncLimiter(max_rate=model_rpm, time_period=60)
    pbar_lock = asyncio.Lock()
    
    async def update_tqdm(task):
        result = await task
        async with pbar_lock:
            pbar.update(1)
        return result
    
    tasks = [
        update_tqdm(generate_CoT(model=model, schemas=schemas, example=example, limiter=limiter if no_limiter_i is None or i >= no_limiter_i else None))
        for i, example in enumerate(decomposer_completer_ds)
    ]

    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    MODEL = 'anthropic/claude-4-sonnet-20250514'
    # MODEL = 'xai/grok-4-0709'
    schemas = DBSchema.from_db_schemas_file(DB_SCHEMAS_JSON_PATH, apply_lower=False)

    dataset = {}
    for split in ['train', 'validation']:
        dataset[split] = asyncio.run(get_examples(model=MODEL, split=split, model_rpm=50, no_limiter_i=10808 if split == 'train' else 1620))
    
    import json
    with open('output/qpl/completer_cot.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    ds = DatasetDict({
        split: Dataset.from_list(data)
        for split, data in dataset.items()
    })
    ds.push_to_hub("d4nieldev/qpl-completer-cot-ds", "original")
