import sys
import json
import random
from typing import List, Dict, Tuple

from inference.qpl.types.schema_types import DBSchema
import utils.qpl.paths as p

import dspy
from sentence_transformers import SentenceTransformer


SEED = 42
LM_ID = "openai/gpt-4o"
SAMPLE_SIZE = 20
FEWSHOT_TRAIN_RATIO = 0.5
FEWSHOT_EXAMPLES = 2
FILENAME = f"predicted_types_concert_singer__{LM_ID.replace('/', '-')}_seed{SEED}_sample{SAMPLE_SIZE}_train_ratio{FEWSHOT_TRAIN_RATIO}_fewshot{FEWSHOT_EXAMPLES}"


random.seed(SEED)

def split_train_test(dataset: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train and test sets."""
    split_index = int(len(dataset) * train_ratio)
    random.shuffle(dataset)
    return dataset[:split_index], dataset[split_index:]


def to_examples(dataset: List[Dict]) -> List[dspy.Example]:
    return [
        dspy.Example(
            database_schema=str(db_schemas[example["db_id"]]),
            entities=[str(entity) for entity in db_schemas[example["db_id"]].entities],
            question=example["question"],
            predicted_type=example["type"]
        ).with_inputs("database_schema", "entities", "question")
        for example in dataset
    ]


class TypeClassificationSignature(dspy.Signature):
    """Predict the type of the result that would be returned by executing an SQL query that correctly answers the question.

The possible types are:
    - {entity} - The result is 1 row that contains {entity}'s primary key (optionally with other columns of {entity}). {entity} should be replaced with one of the entities in the schema.
    - Partial[{entity}] - The result is 1 row that contains multiple columns of {entity} but DOES NOT INCLUDE {entity}'s primary key. {entity} should be replaced with one of the entities in the schema.
    - Reduced[{entity}] - The result is 1 row which is the outcome of a computation on a List[{entity}] or List[Partial[{entity}]]. {entity} should be replaced with one of the entities in the schema.
    - Number - The result is 1 row that contains only a number that is completely unrelated to any entity.
    - Union[{type_1}, {type_2}, ...] - The result is 1 row that is a combination of a subset of the possible types defined above.
    - List[{type}] - if the result returns **multiple** rows, where each row is of type {type}. {type} should be replaced with one of the possible types defined above.
"""
    database_schema: str = dspy.InputField(desc="Database schema described in DDL.")
    entities: List[str] = dspy.InputField(desc="Entities in the schema.")
    question: str = dspy.InputField(desc="Question to be answered by the SQL query.")
    predicted_type: str = dspy.OutputField(desc="Predicted type.")


class TypeClassification(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(
            TypeClassificationSignature,
            rationale_field=dspy.OutputField(desc="Step by step reasoning on the question using the schema and entities to predict the type.")
        )
    
    def forward(self, database_schema: str, entities: List[str], question: str) -> TypeClassificationSignature:
        """Predict the type of the result that would be returned by executing an SQL query that correctly answers the question."""
        return self.generator(database_schema=database_schema, entities=entities, question=question)


# Configure LLM
lm = dspy.LM(LM_ID)
dspy.configure(lm=lm)

# Load database schema from JSON file
db_schemas = DBSchema.from_db_schemas_file(p.DB_SCHEMAS_JSON_PATH)

# Load and prepare the dataset
with open(p.TYPES_CONCERT_SINGER_PATH, "r") as f:
    dataset = json.load(f)
    if SAMPLE_SIZE > 0:
        dataset = random.sample(dataset, k=SAMPLE_SIZE)

train, test = split_train_test(dataset, train_ratio=FEWSHOT_TRAIN_RATIO)
train_examples, test_examples = to_examples(train), to_examples(test)

def type_match(example, pred, trace=None):
    return example.predicted_type.strip().lower() == pred.predicted_type.strip().lower()

st_model = SentenceTransformer("all-MiniLM-L6-v2")
def question_only_embed(texts: list[str]):
    # texts: ["schema: ... | question: ...", ...]
    questions = []
    for txt in texts:
        # Split on " | " and find the "question: " segment
        for part in txt.split(" | "):
            if part.strip().startswith("question:"):
                # Remove the "question: " prefix
                questions.append(part.split("question: ")[1])
                break
        else:
            # Fallback: embed the entire text if "question:" not found
            questions.append(txt)
    return st_model.encode(questions)

fewshot = dspy.KNNFewShot(
    k=FEWSHOT_EXAMPLES,
    trainset=train_examples,
    vectorizer=question_only_embed,  # type: ignore
    
    # few_shot_bootstrap_args
    metric=type_match,
    max_labeled_demos=FEWSHOT_EXAMPLES,
)
compiled_fewshot = fewshot.compile(TypeClassification())

# Predict types
baseline = len(lm.history)
predicted_types = compiled_fewshot.batch(test_examples)
new_calls = len(lm.history) - baseline

# Save logs
sys.stdout = open(p.TYPES_OUTPUT_DIR / f"{FILENAME}.ans", "w")
lm.inspect_history(n=new_calls)
sys.stdout.close()

# Save results
with open(p.TYPES_OUTPUT_DIR / f"{FILENAME}.json", "w") as f:
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