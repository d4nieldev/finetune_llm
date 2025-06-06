import re
import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Literal
from pathlib import Path

from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from inference.qpl.types.schema_types import DBSchema
from inference.qpl.types.test_llm import TypeSystem, TaggedDB, data_path
from custom_types import ChatTemplate, ChatMessage
import utils.qpl.paths as p
from utils.argparse import from_dataclass
from utils.generation import generate_batch, to_model_prompt


@dataclass
class Config:
    """Configuration for the type prediction task."""

    llm_id: str = "Qwen/Qwen3-4B"
    """ID of the LLM to use."""

    db_id: TaggedDB = TaggedDB.CONCERT_SINGER
    """ID of the testing dataset name."""

    seed: int = 42
    """Random seed for reproducibility."""

    filename: Optional[Path] = None
    """Filename for saving the predictions and LLM history (without extension). If None, it will be generated based on the configuration."""

    def __post_init__(self):
        if self.filename is None:
            self.filename = Path(f"hf_pred__model={self.llm_id.replace('/', '-')}_db={self.db_id}_seed={self.seed}")
            
    def to_dict(self) -> Dict[str, str]:
        """Convert the configuration to a dictionary."""
        return {k: str(v) if not isinstance(v, Enum) else v.value for k,v in asdict(self).items()}


def get_chat_template(db_id: str, question: str) -> ChatTemplate:
    """
    Returns a ChatTemplate for the given database ID and question.
    The template includes a system prompt, user prompt, and an empty assistant response.
    """

    db_schemas = DBSchema.from_db_schemas_file(p.DB_SCHEMAS_JSON_PATH)
    schema = db_schemas[db_id]
    
    schema_str = str(schema)
    entities = [str(entity) for entity in schema.entities]


    system_prompt = """Your input fields are:
1. `database_schema` (str): Database schema described in DDL.
2. `entities` (list[str]): Entities in the schema. These are the only allowed entities that can fill the {entity} placeholder in the predicted type.
3. `question` (str): Question to be answered by the SQL query.
Your output fields are:
1. `predicted_type` (str): Predicted type.
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## database_schema ## ]]
{database_schema}

[[ ## entities ## ]]
{entities}

[[ ## question ## ]]
{question}

[[ ## predicted_type ## ]]
{predicted_type}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Predict the type of the result that would be returned by executing an SQL query that correctly answers the question.
        
        The possible types are:
            - {entity} - The result columns are all from {entity}. In case a only a foreign key is returned from a table, the **referred** entity is what counts. {entity} should be replaced with one of the entities in the schema.
            - Aggregated[{entity}] - The result is the outcome of a computation derived from a stream of {entity}s without additional columns. {entity} should be replaced with one of the entities in the schema.
            - {type_1}, {type_2}, ..., {type_n} - The result is a combination of {entity}s and Aggregated[{entity}]s (not necessarily the same {entity})."""

    user_prompt = f"""[[ ## database_schema ## ]]
{schema_str}

[[ ## entities ## ]]
{entities}

[[ ## question ## ]]
{question}

Respond with the corresponding output fields, starting with the field `[[ ## predicted_type ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."""
    
    return ChatTemplate(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
    )


def parse_args() -> Config:
    """Parse command line arguments and return a Config object."""
    return from_dataclass(Config)


def main(config: Config):
    print(f"Configuration: {config.to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(config.llm_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.llm_id,
        torch_dtype="auto",
        device_map="auto"
    )

    dataset = json.loads(data_path(TypeSystem.SIMPLE, config.db_id).read_text())
    chat_templates = map(lambda x: get_chat_template(config.db_id.value, x["question"]), dataset)
    model_prompts =list(map(lambda ct: to_model_prompt(tokenizer, ct, enable_thinking=True), chat_templates))

    def extract_fields(model_output: str | None) -> Dict[str, str]:
        # reasoning_idx = model_output.index("[[ ## reasoning ## ]]")
        if model_output is None:
            return {
                'reasoning': "",
                'predicted_type': "",
            }
        reasoning = re.findall(f'<think>(.*?)</think>', model_output, re.DOTALL)
        predicted_type_idx = model_output.index("[[ ## predicted_type ## ]]")
        completed_idx = model_output.index("[[ ## completed ## ]]")
        
        return {
            # 'reasoning': model_output[reasoning_idx + len("[[ ## reasoning ## ]]"):predicted_type_idx].strip(),
            'reasoning': reasoning[0].strip() if reasoning else "",
            'predicted_type': model_output[predicted_type_idx + len("[[ ## predicted_type ## ]]"):completed_idx].strip(),
        }

    outputs = generate_batch(
        model=model,
        tokenizer=tokenizer,
        model_prompts=model_prompts,
        batch_size=8,
        max_new_tokens=4096,
        progress_bar=tqdm(total=len(model_prompts), desc="Predicting types"),
        is_valid_output=lambda idx, output: all(bool(val) for val in extract_fields(output).values()),
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
    )
    
    with open(p.TYPES_OUTPUT_DIR / f"{args.filename}.json", "w") as f:
        json.dump(
            [
                {
                    "db_id": example["db_id"],
                    "question": example["question"],
                    **extract_fields(generation),
                    "actual_type": example["type"],
                }
                for example, generation in zip(dataset, outputs)
            ],
            f,
            indent=2
        )
    
    print(f"Saved predictions to {p.TYPES_OUTPUT_DIR / f'{args.filename}.json'}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    

