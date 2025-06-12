import sys
import json
import random
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from abc import ABC

import dspy
from sentence_transformers import SentenceTransformer

from src.inference.qpl.types.schema_types import DBSchema
from src.utils.lists import split_train_test
import src.utils.qpl.paths as p
from src.utils.argparse import from_dataclass


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(pathname)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


class TaggedDB(Enum):
    CONCERT_SINGER = "concert_singer"
    BATTLE_DEATH = "battle_death"
    
class TypeSystem(Enum):
    VERBOSE = "verbose"
    SIMPLE = "simple"
    AUTO = "auto"

    @property
    def signature(self):
        """Return the signature of the type system."""
        return {
            TypeSystem.VERBOSE: VerboseTypeSystem,
            TypeSystem.SIMPLE: SimpleTypeSystem,
            TypeSystem.AUTO: AutoTypeSystem,
        }[self]
    
REASONING_MODELS = ['qwen3']
    

def data_path(type_system: TypeSystem, db_id: TaggedDB) -> Path:
    """Return the path to the dataset file based on the database ID and type system."""
    return p.TYPES_INPUT_DIR / f"{type_system.value}_{db_id.value}.json"


@dataclass
class Config:
    """Configuration for the type prediction task."""

    llm_id: str = "ollama_chat/qwen3:4b"
    """ID of the LLM to use."""

    type_system: TypeSystem = TypeSystem.AUTO
    """Type system to use for type prediction."""

    train_db_id: TaggedDB = TaggedDB.BATTLE_DEATH
    """ID of the training dataset (examples for fewshot) name."""

    test_db_id: TaggedDB = TaggedDB.BATTLE_DEATH
    """ID of the testing dataset name."""

    split_train_ratio: float = 0
    """Ratio of training data to use - in case train and test datasets are the same."""

    fewshot_examples: int = 0
    """Number of few-shot examples to use."""

    cot: bool = True
    """Whether to use Chain of Thought reasoning."""

    test_sample_size: int = -1
    """Sample size for testing dataset. -1 means no sampling."""

    train_sample_size: int = -1
    """Sample size for training dataset. -1 means no sampling."""

    seed: int = 42
    """Random seed for reproducibility."""

    filename: Optional[Path] = None
    """Filename for saving the predictions and LLM history (without extension). If None, it will be generated based on the configuration."""

    def __post_init__(self):
        if self.filename is None:
            if self.train_db_id != self.test_db_id:
                data_str = f"train={self.train_db_id.value}_test={self.test_db_id.value}"
            else:
                data_str = f"dataset={self.train_db_id.value}_split_train_ratio={self.split_train_ratio}"
            sample_str = f"_train_sample_size={self.train_sample_size}" if self.train_sample_size > 0 else ""
            sample_str += f"_test_sample_size={self.test_sample_size}" if self.test_sample_size > 0 else ""
            self.filename = Path(f"pred__model={self.llm_id.replace('/', '-')}_system={self.type_system.value}_{data_str}{sample_str}_fewshot={self.fewshot_examples}{'_cot' if self.cot else ''}_seed={self.seed}")
        
        if self.train_db_id != self.test_db_id and self.split_train_ratio > 0:
            self.split_train_ratio = 0
            log.info("Using different train and test datasets, setting split_train_ratio to 0.")
        
        if self.fewshot_examples == 0:
            if self.train_db_id != self.test_db_id or self.split_train_ratio > 0:
                log.info("Using no few-shot examples, but train and test datasets are different or split_train_ratio is greater than 0. This may lead to unexpected results.")
            self.train_db_id = self.test_db_id
            self.split_train_ratio = 0

        if self.fewshot_examples < 0:
            raise ValueError("fewshot_examples must be a non-negative integer.")

        if self.train_db_id == self.test_db_id:
            if self.train_sample_size != self.test_sample_size:
                raise ValueError("If train and test datasets are the same, train_sample_size and test_sample_size must be equal.")
            if self.fewshot_examples > 0 and self.split_train_ratio == 0:
                raise ValueError("If fewshot_examples > 0, split_train_ratio must be greater than 0.")
            
    def to_dict(self) -> Dict[str, str]:
        """Convert the configuration to a dictionary."""
        return {k: str(v) if not isinstance(v, Enum) else v.value for k,v in asdict(self).items()}
    

class TypeSystemFields(dspy.Signature, ABC):
    database_schema: str = dspy.InputField(desc="Database schema described in DDL.")
    entities: List[str] = dspy.InputField(desc="Entities in the schema. These are the only allowed entities that can fill the {entity} placeholder in the predicted type.")
    question: str = dspy.InputField(desc="Question to be answered by the SQL query.")
    predicted_type: str = dspy.OutputField(desc="Predicted type.")

class VerboseTypeSystem(TypeSystemFields):
    """Predict the type of the result that would be returned by executing an SQL query that correctly answers the question.

The possible types are:
    - PK[{entity}] - The result is 1 row that contains the primary key (or a reference to a primary key) of {entity} optionally with additional column(s) of {entity}. {entity} should be replaced with one of the entities in the schema.
    - NoPK[{entity}] - The result is 1 row that contains column(s) of {entity} - NOT INCLUDING its primary key. {entity} should be replaced with one of the entities in the schema.
    - Aggregated[{entity}] - The result is 1 row which is the outcome of a computation derived from a stream of {entity}s. {entity} should be replaced with one of the entities in the schema.
    - Number - The result is 1 row that contains a single number that is not derived from any entity.
    - Union[{type_1}, {type_2}, ...] - The result is 1 row that is defined by {type_1}, {type_2}, ... which are a subset of the possible types defined above.
    - List[{type}] - The result is a stream of rows, where each row is of type {type}. {type} should be replaced with one of the possible types defined above.
"""


class SimpleTypeSystem(TypeSystemFields):
    """Predict the type of the result that would be returned by executing an SQL query that correctly answers the question.

The possible types are:
    - {entity} - The result columns are all from {entity}. In case only a foreign key is returned from a table, the **referred** entity is what counts. {entity} should be replaced with one of the entities in the schema.
    - Aggregated[{entity}] - The result is the outcome of a computation derived from a stream of {entity}s without additional columns. {entity} should be replaced with one of the entities in the schema.
    - {type_1}, {type_2}, ..., {type_n} - The result is a combination of {entity}s and Aggregated[{entity}]s (not necessarily the same {entity}).
"""


class AutoTypeSystem(TypeSystemFields):
    """Predict the type of the result that would be returned by executing an SQL query that correctly answers the question.

The possible types are:
    - {entity} - The result contains columns from {entity}. In case a foreign key is in the result, the **referred** entity that **originally** contains this column (NOT as a foreign key) is what counts. {entity} should be replaced with one of the entities in the schema.
    - Aggregated[{entity}] - The result contains at least one column that is the outcome of a computation derived from a stream of {entity}s and this outcome is a value of one of the rows in the table (for example MIN/MAX). {entity} should be replaced with one of the entities in the schema.
    - Number - The result is a number that is either (1) a computation derived from a steam of entities, and this outcome is NOT a value of one of the rows in the table (for example COUNT/AVG/SUM), or (2) a number that is not derived from any entity.
    - {type_1}, {type_2}, ..., {type_n} - The result is a combination of {entity}s and Aggregated[{entity}]s (not necessarily the same {entity}).
"""


class TypeClassifier(dspy.Module):
    def __init__(self, cot: bool, type_system_signature: type[dspy.Signature]):
        super().__init__()
        if cot:
            self.generator = dspy.ChainOfThought(
                type_system_signature,
                rationale_field=dspy.OutputField(desc="Step by step reasoning on the question using the schema and entities to predict the type.")
            )
        else:
            self.generator = dspy.Predict(type_system_signature)
        
    def forward(self, database_schema: str, entities: List[str], question: str):
        """Predict the type of the result that would be returned by executing an SQL query that correctly answers the question."""
        return self.generator(database_schema=database_schema, entities=entities, question=question)
    

def to_examples(dataset: List[Dict], db_schemas: Dict[str, DBSchema]) -> List[dspy.Example]:
    return [
        dspy.Example(
            database_schema=str(db_schemas[example["db_id"]]),
            entities=[str(entity) for entity in db_schemas[example["db_id"]].entities],
            question=example["question"],
            predicted_type=example["type"]
        ).with_inputs("database_schema", "entities", "question")
        for example in dataset
    ]
    

def load_dataset(
        type_system: TypeSystem,
        train_db: TaggedDB,
        test_db: TaggedDB,
        train_sample_size: int,
        test_sample_size: int,
        split_train_ratio: float
    ) -> Tuple[List[Dict], List[Dict]]:

    if train_db == test_db:
        dataset = json.loads(data_path(type_system, train_db).read_text())
        if test_sample_size > 0:
            dataset = random.sample(dataset, k=test_sample_size)
        train, test = split_train_test(dataset, train_ratio=split_train_ratio)
    else:
        train = json.loads(data_path(type_system, train_db).read_text())
        if train_sample_size > 0:
            train = random.sample(train, k=train_sample_size)
        test = json.loads(data_path(type_system, test_db).read_text())
        if test_sample_size > 0:
            test = random.sample(test, k=test_sample_size)
    
    return train, test


def compile_knn_fewshot(
        train_examples: List[dspy.Example],
        num_fewshot_examples: int,
        cot: bool,
        type_system_signature: type[dspy.Signature]
    ) -> dspy.Module:
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def question_only_embed(texts: list[str]):
        questions = []
        for txt in texts:
            for part in txt.split(" | "):
                if part.strip().startswith("question:"):
                    questions.append(part.split("question: ")[1])
                    break
            else:
                questions.append(txt)
        return st_model.encode(questions)
    
    def type_match(example, pred, trace=None):
        return example.predicted_type.strip().lower() == pred.predicted_type.strip().lower()

    fewshot = dspy.KNNFewShot(
        k=num_fewshot_examples,
        trainset=train_examples,
        vectorizer=question_only_embed,  # type: ignore
        metric=type_match,
    )

    program = fewshot.compile(TypeClassifier(cot=cot, type_system_signature=type_system_signature))

    return program


def parse_args() -> Config:
    args = from_dataclass(Config)
    if args.cot and any([model_id in args.llm_id for model_id in REASONING_MODELS]):
        log.warning(f"Model {args.llm_id!r} has built-in reasoning, I recommend setting `cot=False` to prevent overthinking.")
    log.info(f"Using Config:\n{json.dumps(args.to_dict(), indent=2)}")
    return args


def main(args: Config):
    # Configure LLM
    if args.llm_id.startswith("ollama_chat/"):
        lm = dspy.LM(
            args.llm_id,
            api_base="http://localhost:11434",
            api_key=""
        )
    else:
        lm = dspy.LM(args.llm_id)
    
    dspy.configure(lm=lm)
    random.seed(args.seed)

    # Load data
    train, test = load_dataset(
        type_system=args.type_system,
        train_db=args.train_db_id,
        test_db=args.test_db_id,
        train_sample_size=args.train_sample_size,
        test_sample_size=args.test_sample_size,
        split_train_ratio=args.split_train_ratio
    )
    db_schemas = DBSchema.from_db_schemas_file(p.DB_SCHEMAS_JSON_PATH)
    train_examples, test_examples = to_examples(train, db_schemas), to_examples(test, db_schemas)

    # Compile program
    if args.fewshot_examples > 0 and len(train_examples) > 0:
        program = compile_knn_fewshot(
            train_examples=train_examples,
            num_fewshot_examples=args.fewshot_examples,
            cot=args.cot,
            type_system_signature=args.type_system.signature
        )
    else:
        program = TypeClassifier(cot=args.cot, type_system_signature=args.type_system.signature)

    # Predict types
    baseline = len(lm.history)
    predicted_types = program.batch(test_examples)
    new_calls = len(lm.history) - baseline

    # Save LLM history
    sys.stdout = open(p.TYPES_OUTPUT_DIR / f"{args.filename}.ans", "w")
    lm.inspect_history(n=new_calls)
    sys.stdout.close()
    log.info(f"Saved LLM history to {p.TYPES_OUTPUT_DIR / f'{args.filename}.ans'}")

    # Save results
    with open(p.TYPES_OUTPUT_DIR / f"{args.filename}.json", "w") as f:
        json.dump(
            [
                {
                    "db_id": example["db_id"],
                    "question": example["question"],
                    "reasoning": pt.reasoning if hasattr(pt, 'reasoning') else None,
                    "predicted_type": pt.predicted_type,
                    "actual_type": example["type"],
                }
                for example, pt in zip(test, predicted_types)
            ],
            f, 
            indent=2
        )
    log.info(f"Saved predictions to {p.TYPES_OUTPUT_DIR / f'{args.filename}.json'}")


if __name__ == "__main__":
    args = parse_args()
    main(args)