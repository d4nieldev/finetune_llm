import json
import dspy
from typing import List

from inference.qpl.types.schema_types import DBSchema


class TypeClassification(dspy.Signature):
    """Predict the type of the result that would be returned by executing an SQL query that correctly answers the question.

The possible types are:
    - {entity} - The result is a row that contains multiple columns of an entity entity including {entity}'s primary key. {entity} should be replaced with one of the entities in the schema.
    - Partial[{entity}] - The result is a row that contains multiple columns of an entity entity but does not include {entity}'s primary key. {entity} should be replaced with one of the entities in the schema.
    - Reduced[{entity}] - The result is the outcome of a computation on a List[{entity}] or List[Partial[{entity}]]. {entity} should be replaced with one of the entities in the schema.
    - Number - The result is just a number that is not the result of a calculation on a List of entities.
    - Union[{type_1}, {type_2}, ..., {type_n}] - The result is a row that is a combination of a subset of the possible types defined above..
    - List[{type}] - if the result returns a list of any of the types defined above. {type} should be replaced with one of the possible types.
"""
    database_schema: str = dspy.InputField(desc="Database schema described in DDL.")
    entities: List[str] = dspy.InputField(desc="Entities in the schema.")
    question: str = dspy.InputField(desc="Question to be answered by the SQL query.")
    predicted_type: str = dspy.OutputField(desc="Predicted type.")


# Configure LLM
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

# Load database schema from JSON file
with open("data/qpl/spider/db_schemas.json", "r") as f:
    db_schemas_json = json.load(f)
db_schemas = DBSchema.from_db_schemas(db_schemas_json)

# Configure prompt
pred = dspy.Predict(TypeClassification)
schema = db_schemas["concert_singer"]
database_schema = str(schema)
entities = ", ".join([str(entity) for entity in schema.entities])
question = "Show name, country, age for all singers ordered by age from the oldest to the youngest."

# Predict type
predicted_type = pred(
    database_schema=database_schema,
    entities=entities,
    question=question
).predicted_type

# Show prompt
print(lm.inspect_history(1))