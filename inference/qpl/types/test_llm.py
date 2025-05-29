import dspy


class TypeClassification(dspy.Signature):
    """
    Predict the type of the result that would be returned by executing an SQL query that correctly answers the question.

    The possible types are:
        - {entity} - The result is a row that contains multiple columns of an entity entity including {entity}'s primary key. {entity} should be replaced with one of the entities in the schema.
	    - Partial[{entity}] - The result is a row that contains multiple columns of an entity entity but does not include {entity}'s primary key. {entity} should be replaced with one of the entities in the schema.
	    - Reduced[{entity}] - The result is the outcome of a computation on a List[{entity}] or List[Partial[{entity}]]. {entity} should be replaced with one of the entities in the schema.
	    - Number - The result is just a number that is not the result of a calculation on a List of entities.
	    - Union[{type_1}, {type_2}, ..., {type_n}] - The result is a row that is a combination of a subset of the possible types defined above..
        - List[{type}] - if the result returns a list of any of the types defined above. {type} should be replaced with one of the possible types.
    """
    database_schema: str = dspy.InputField(desc="Database schema described in DDL.")
    entities: str = dspy.InputField(desc="Entities in the schema, separated by commas.")
    question: str = dspy.InputField(desc="Question to be answered by the SQL query.")
    predicted_type: str = dspy.OutputField(desc="Predicted type.")
