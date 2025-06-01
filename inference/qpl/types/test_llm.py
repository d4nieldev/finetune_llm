import sys
import json
import random
import logging as log
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from inference.qpl.types.schema_types import DBSchema
from utils.lists import split_train_test
import utils.qpl.paths as p
from utils.argparse import from_dataclass

import dspy
from sentence_transformers import SentenceTransformer


class TaggedDB(Enum):
    CONCERT_SINGER = "concert_singer"
    BATTLE_DEATH = "battle_death"

    @property
    def path(self) -> Path:
        """Return the path to the dataset file."""
        return {
            TaggedDB.CONCERT_SINGER: p.TYPES_CONCERT_SINGER_PATH,
            TaggedDB.BATTLE_DEATH: p.TYPES_BATTLE_DEATH_PATH,
        }[self]


@dataclass
class Config:
    """Configuration for the type prediction task."""

    llm_id: str = "openai/gpt-4.1-mini"
    """ID of the LLM to use."""

    train_db_id: TaggedDB = TaggedDB.BATTLE_DEATH
    """ID of the training dataset (examples for fewshot) name."""

    test_db_id: TaggedDB = TaggedDB.CONCERT_SINGER
    """ID of the testing dataset name."""

    split_train_ratio: Optional[float] = None
    """Ratio of training data to use - in case train and test datasets are the same."""

    fewshot_examples: int = 3
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
                f"dataset={self.train_db_id.value}_split_train_ratio={self.split_train_ratio}"
            sample_str = f"_train_sample_size={self.train_sample_size}" if self.train_sample_size > 0 else ""
            sample_str += f"_test_sample_size={self.test_sample_size}" if self.test_sample_size > 0 else ""
            self.filename = Path(f"pred_{self.llm_id.replace('/', '-')}_{data_str}{sample_str}_fewshot={self.fewshot_examples}{'_cot' if self.cot else ''}_seed={self.seed}")
        
        if self.fewshot_examples < 0:
            raise ValueError("fewshot_examples must be a non-negative integer.")

        if self.train_db_id == self.test_db_id and self.train_sample_size != self.test_sample_size:
            raise ValueError("If train and test datasets are the same, train_sample_size and test_sample_size must be equal.")


class VerboseTypeSystem(dspy.Signature):
    """Predict the type of the result that would be returned by executing an SQL query that correctly answers the question.

The possible types are:
    - PK[{entity}] - The result is 1 row that contains the primary key (or a reference to a primary key) of {entity} optionally with additional column(s) of {entity}. {entity} should be replaced with one of the entities in the schema.
    - NoPK[{entity}] - The result is 1 row that contains column(s) of {entity} - NOT INCLUDING its primary key. {entity} should be replaced with one of the entities in the schema.
    - Aggregated[{entity}] - The result is 1 row which is the outcome of a computation derived from a stream of {entity}s. {entity} should be replaced with one of the entities in the schema.
    - Number - The result is 1 row that contains a single number that is not derived from any entity.
    - Union[{type_1}, {type_2}, ...] - The result is 1 row that is defined by {type_1}, {type_2}, ... which are a subset of the possible types defined above.
    - List[{type}] - The result is a stream of rows, where each row is of type {type}. {type} should be replaced with one of the possible types defined above.
"""
    database_schema: str = dspy.InputField(desc="Database schema described in DDL.")
    entities: List[str] = dspy.InputField(desc="Entities in the schema.")
    question: str = dspy.InputField(desc="Question to be answered by the SQL query.")
    predicted_type: str = dspy.OutputField(desc="Predicted type.")


class TypeClassifier(dspy.Module):
    def __init__(self, cot: bool):
        super().__init__()
        if cot:
            self.generator = dspy.ChainOfThought(
                VerboseTypeSystem,
                rationale_field=dspy.OutputField(desc="Step by step reasoning on the question using the schema and entities to predict the type.")
            )
        else:
            self.generator = dspy.Predict(VerboseTypeSystem)
        
    def forward(self, database_schema: str, entities: List[str], question: str) -> VerboseTypeSystem:
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
        train_db: TaggedDB,
        test_db: TaggedDB,
        train_sample_size: int,
        test_sample_size: int,
        split_train_ratio: Optional[float]
    ) -> Tuple[List[Dict], List[Dict]]:

    if train_db == test_db:
        if not split_train_ratio:
            raise ValueError("If train and test datasets are the same, split_train_ratio must be provided.")
        
        dataset = json.loads(train_db.path.read_text())
        if test_sample_size > 0:
            dataset = random.sample(dataset, k=test_sample_size)
        train, test = split_train_test(dataset, train_ratio=split_train_ratio)
    else:
        train = json.loads(train_db.path.read_text())
        if train_sample_size > 0:
            train = random.sample(train, k=train_sample_size)
        test = json.loads(test_db.path.read_text())
        if test_sample_size > 0:
            test = random.sample(test, k=test_sample_size)
    
    return train, test


def compile_knn_fewshot(train_examples: List[dspy.Example], num_fewshot_examples: int, cot: bool) -> dspy.Module:
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

    program = fewshot.compile(TypeClassifier(cot=cot))

    return program


def parse_args() -> Config:
    return from_dataclass(Config)


def main(args: Config):
    # Configure LLM
    lm = dspy.LM(args.llm_id)
    dspy.configure(lm=lm)
    random.seed(args.seed)

    # Load data
    train, test = load_dataset(
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
        program = compile_knn_fewshot(train_examples=train_examples, num_fewshot_examples=args.fewshot_examples, cot=args.cot)
    else:
        program = TypeClassifier(cot=args.cot)

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
                    "actual_type": example["type"],
                    "reasoning": pt.reasoning,
                    "predicted_type": pt.predicted_type,
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