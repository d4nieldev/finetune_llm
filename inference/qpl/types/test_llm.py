import sys
import json
import random
from typing import List, Dict, Tuple

from inference.qpl.types.schema_types import DBSchema
import utils.qpl.paths as p

import dspy

SEED = 42
LM_ID = "openai/gpt-4o-mini"
SAMPLE_SIZE = 10
FEWSHOT_TRAIN_RATIO = 0.5
FEWSHOT_EXAMPLES = 2
FILENAME = f"predicted_types_concert_singer_seed{SEED}_sample{SAMPLE_SIZE}_train_ratio{FEWSHOT_TRAIN_RATIO}_fewshot{FEWSHOT_EXAMPLES}"


random.seed(SEED)

def split_train_test(dataset: List[Dict], train_ratio: float = 0.8) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Split dataset into train and test sets."""
    split_index = int(len(dataset) * train_ratio)
    inputs_dataset = [
        dspy.Example(
            database_schema=str(db_schemas[example["db_id"]]),
            entities=[str(entity) for entity in db_schemas[example["db_id"]].entities],
            question=example["question"],
            predicted_type=example["type"]
        ).with_inputs("database_schema", "entities", "question")
        for example in dataset
    ]
    random.shuffle(dataset)
    return inputs_dataset[:split_index], inputs_dataset[split_index:]


class TypeClassification(dspy.Signature):
    """Predict the type of the result that would be returned by executing an SQL query that correctly answers the question.

The possible types are:
    - {entity} - The result is a row that contains {entity}'s primary key (optionally with other columns of {entity}). {entity} should be replaced with one of the entities in the schema.
    - Partial[{entity}] - The result is a row that contains multiple columns of an entity entity but DOES NOT INCLUDE {entity}'s primary key. {entity} should be replaced with one of the entities in the schema.
    - Reduced[{entity}] - The result is the outcome of a computation on a List[{entity}] or List[Partial[{entity}]]. {entity} should be replaced with one of the entities in the schema.
    - Number - The result is just a number that is not the result of a calculation on a List of entities.
    - Union[{type_1}, {type_2}, ..., {type_n}] - The result is a row that is a combination of a subset of the possible types defined above.
    - List[{type}] - if the result returns multiple rows, where each row is of type {type}. {type} should be replaced with one of the possible types defined above.
"""
    database_schema: str = dspy.InputField(desc="Database schema described in DDL.")
    entities: List[str] = dspy.InputField(desc="Entities in the schema.")
    question: str = dspy.InputField(desc="Question to be answered by the SQL query.")
    predicted_type: str = dspy.OutputField(desc="Predicted type.")


class TypeClassificationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(TypeClassification)
    
    def forward(self, database_schema: str, entities: List[str], question: str) -> TypeClassification:
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

generate_type = TypeClassificationModule()
fewshot = dspy.LabeledFewShot(k=FEWSHOT_EXAMPLES)
compiled_fewshot = fewshot.compile(
    generate_type,
    trainset=train,
)

# Predict types
predicted_types = compiled_fewshot.batch(test)

# Save logs
sys.stdout = open(p.TYPES_OUTPUT_DIR / f"{FILENAME}.txt", "w")
lm.inspect_history(len(predicted_types))
sys.stdout.close()

# Save results
with open(p.TYPES_OUTPUT_DIR / f"{FILENAME}.json", "w") as f:
    json.dump(
        [
            {
                "db_id": example["db_id"],
                "question": example["question"],
                "actual_type": example["type"],
                "predicted_type": pt.predicted_type,
                "reasoning": pt.reasoning if hasattr(pt, 'reasoning') else None
            }
            for example, pt in zip(dataset, predicted_types)
        ],
        f, 
        indent=2
    )