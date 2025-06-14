from typing import Set, Optional, List, Tuple
from enum import StrEnum
from collections import defaultdict

from src.utils.qpl.tree import Operator, QPLTree
from src.utils.qpl.paths import DB_SCHEMAS_JSON_PATH
from src.inference.qpl.types.qpl_to_type import QPLType, NUMBER, qpl_tree_to_type
from src.inference.qpl.types.schema_types import DBSchema
from src.utils.lists import powerset


def set_str(s: Set, wrap: bool = True) -> str:
    output = ', '.join(sorted(str(item) for item in s)) if s else 'empty set'
    if wrap:
        return f"'{output}'"
    return output


class Status(StrEnum):
    SUCCESS = "Success"
    INFERENCE_ERROR = "Inference Error"
    OPERATOR_ERROR = "Operator Error"
    STRUCTURE_ERROR = "Structure Error"
    UNEXPECTED_ERROR = "Unexpected Error"


class IncompatibleTypesError(Exception):
    def __init__(self, op: Operator, type_1: Set[QPLType], type_2: Optional[Set[QPLType]]):
        if type_2 is None:
            type_2 = set()
        super().__init__(f"Incompatible types for operator {op.value!r}: {set_str(type_1)} and {set_str(type_2)}")
        self.op = op
        self.type_1 = type_1
        self.type_2 = type_2


def check_and_resolve(op: Operator, type_1: Set[QPLType], type_2: Optional[Set[QPLType]] = None) -> List[Set[QPLType]]:
    """
    Check if the input type(s) are compatible with the operator and return a list of possible output types.
    If the types are incompatible, raise an IncompatibleTypesError.
    """
    if op == Operator.AGGREGATE:
        # input_type = (t1, t2)
        # output_types = [(t1), (t2), (Aggregated[t1]), (Aggregated[t2]),
        #                 (t1, t2), (t1, Aggregated[t2]), (Aggregated[t1], t2), (Aggregated[t1], Aggregated[t2])
        #                 (t1, t2, Aggregated[t1]), (t1, t2, Aggregated[t2]), (t1, t2, Aggregated[t1], Aggregated[t2])]
        # each Aggregated can also be a Number, or in addition to Number on the same entity
        types = [
            set(agg_type)
            for candidates in powerset(type_1, include_empty=False)
            for agg_type in powerset({t for c in candidates for t in [c, QPLType(c.entity, aggregated=True), QPLType(NUMBER)]}, include_empty=False)
        ]
    elif op in [Operator.FILTER, Operator.TOP, Operator.SORT, Operator.TOPSORT]:
        types = [set(sub_t) for sub_t in powerset(type_1, include_empty=False)]  # any subset of type_1 is valid, except empty set
    elif op == Operator.JOIN:
        if not type_2:
            raise ValueError("Join operator requires two types.")
        
        # Number < Aggregated[Entity] < Entity
        interscection = set(
            max(t1, t2, key=lambda t: 0 if t.entity == NUMBER else 1 if t.aggregated else 2)
            for t1 in type_1
            for t2 in type_2
            if t1.entity == t2.entity or (t1.entity is None or t2.entity is None)
        )
        
        if not interscection:
            raise IncompatibleTypesError(op, type_1, type_2)
        
        types = [set(t) for t in powerset(type_1.union(type_2), include_empty=False)]
    elif op in [Operator.EXCEPT, Operator.INTERSECT, Operator.UNION]:
        # FIXME: this is a heuristic, for better type checking, we should document the number of columns returned from each child
        #                             for even better type checking, we should also include the column types or return the actual column names
        if not type_2:
            raise ValueError(f"{op.value} operator requires two types.")
        
        intersection = type_1.intersection(type_2)
        if not intersection:
            raise IncompatibleTypesError(op, type_1, type_2)
        
        # any subset of the intersection between type_1 and type_2 is valid, except empty set
        types = [set(sub_t) for sub_t in powerset(intersection, include_empty=False)]
    else:
        raise ValueError(f"Unknown operator: {op!r}.")
    
    if {QPLType(NUMBER)} not in types:
        # Number is always a valid type because the question can be "List 1 for ..."
        types.append({QPLType(NUMBER)})
    
    return [t for i, t in enumerate(types) if t not in types[:i]]  # remove duplicates 
    

def rec_type_check(tree: QPLTree, schema: DBSchema) -> Tuple[Status, str]:
    for child in tree.children:
        status, error_message = rec_type_check(child, schema)
        if status != Status.SUCCESS:
            return status, error_message
    try:
        root_type = qpl_tree_to_type(tree, schema).type_set
        children_types = [qpl_tree_to_type(child, schema).type_set for child in tree.children]
    except AssertionError as e:
        return Status.INFERENCE_ERROR, f"TypeInferenceError in line #{tree.line_num}: {str(e)}"
    except Exception as e:
        return Status.UNEXPECTED_ERROR, f"UnexpectedTypeInference error in line #{tree.line_num}: {str(e)}"

    if not children_types:
        return Status.SUCCESS, "Success."
    try:
        possible_types = check_and_resolve(tree.op, *children_types)
        if root_type in possible_types:
            return Status.SUCCESS, "Success."
        return Status.STRUCTURE_ERROR, f"StructureError in line #{tree.line_num}: Input types {' and '.join([set_str(child) for child in children_types])} are incompatible with output type {set_str(root_type)} for operator {tree.op.value!r}. Possible types: {[set_str(pt, wrap=False) for pt in possible_types]}."
    except IncompatibleTypesError as e:
        return Status.OPERATOR_ERROR, f"OperatorError in line #{tree.line_num}: {str(e)}"
    

if __name__ == "__main__":
    import json
    from datasets import load_dataset

    dataset = load_dataset("d4nieldev/nl2qpl-ds")
    success = 0
    total = 0
    errors = []
    err_counts = defaultdict(int)

    schemas = DBSchema.from_db_schemas_file(DB_SCHEMAS_JSON_PATH)

    for split in dataset:
        for row in dataset[split]:
            qpl = row['qpl']
            db_id, qpl = qpl.split(' | ')

            schema = schemas[db_id]
            tree = QPLTree.from_qpl_lines(qpl.split(' ; '))

            status, error = rec_type_check(tree=tree, schema=schema)
            if status == Status.SUCCESS:
                success += 1
            else:
                errors.append({"split": split, "db_id": db_id, **tree.to_dict(), "status": status, "error": error})
                err_counts[status.value] += 1
            
            total += 1
    
    print(f"Type check success rate: {success / total:.2%} ({success}/{total})")
    print(f"Errors breakdown: {dict(err_counts)}")

    with open("type_check_errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    print(f"Type check errors saved to type_check_errors.json")

