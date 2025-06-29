from typing import Set, Optional, List, Tuple, Dict, Protocol
from enum import StrEnum
from collections import defaultdict, Counter

from src.utils.qpl.tree import Operator, QPLTree
from src.utils.qpl.paths import DB_SCHEMAS_JSON_PATH
from src.inference.qpl.types.qpl_to_type import QPLType, NUMBER, qpl_tree_to_type, types_str
from src.inference.qpl.types.schema_types import DBSchema
from src.utils.lists import powerset


class Status(StrEnum):
    SUCCESS = "Success"
    SYNTAX_ERROR = "Syntax Error"
    OPERATOR_ERROR = "Operator Error"
    STRUCTURE_ERROR = "Structure Error"
    UNEXPECTED_ERROR = "Unexpected Error"


class TypeValidator(Protocol):
    def __call__(self, type_cnt: Dict[QPLType, int]) -> bool:
        """
        Validate the type count dictionary.
        Returns True if valid, False otherwise.
        """
        ...


class IncompatibleTypesError(Exception):
    def __init__(self, op: Operator, type_1: Dict[QPLType, int], type_2: Optional[Dict[QPLType, int]], suffix: str = ""):
        msg =  f"Incompatible types for operator {op.value!r}: {types_str(type_1)!r}"
        if type_2 is not None:
            msg += f" and {types_str(type_2)!r}"
        super().__init__(msg + suffix)
        self.op = op
        self.type_1 = type_1
        self.type_2 = type_2


def check_and_resolve(op: Operator, type_1: Dict[QPLType, int], type_2: Optional[Dict[QPLType, int]] = None, strict: bool = True) -> TypeValidator:
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
        types_sets = [
            frozenset(agg_type)
            for candidates in powerset(type_1.keys(), include_empty=False)
            for agg_type in powerset({t for c in candidates for t in {c, QPLType(c.entity, aggregated=True), QPLType(NUMBER)}}, include_empty=False)
        ]
    elif op in [Operator.FILTER, Operator.TOP, Operator.SORT, Operator.TOPSORT]:
        types_sets = [frozenset(sub_t) for sub_t in powerset(type_1.keys(), include_empty=False)]  # any subset of type_1 is valid, except empty set
    elif op == Operator.JOIN:
        if not type_2:
            raise ValueError("Join operator requires two types.")
        
        # Number < Aggregated[Entity] < Entity & Number can be joined with any other type
        interscection = set(
            max(t1, t2, key=lambda t: 0 if t.entity == NUMBER else 1 if t.aggregated else 2)
            for t1 in type_1.keys()
            for t2 in type_2.keys()
            if t1.entity == t2.entity or t1.entity == NUMBER or t2.entity == NUMBER
        )
        
        if not interscection:
            raise IncompatibleTypesError(op, type_1, type_2)
        
        types_sets = [frozenset(t) for t in powerset(set(type_1.keys()).union(set(type_2.keys())), include_empty=False)]
    elif op in [Operator.EXCEPT, Operator.INTERSECT, Operator.UNION]:
        if not type_2:
            raise ValueError(f"{op.value} operator requires two types.")
        
        if strict and sum(type_1.values()) != sum(type_2.values()):
            raise IncompatibleTypesError(op, type_1, type_2, suffix=f" (must have the same number of columns in total: {sum(type_1.values())} != {sum(type_2.values())}).")

        input_types = set(type_1.keys()).union(set(type_2.keys()))
        types_cnt = {t: max(type_1.get(t, 0), type_2.get(t, 0)) for t in input_types}
        return lambda type_cnt: type_cnt == {QPLType(NUMBER): 1} or all(type_cnt.get(t, 0) <= types_cnt.get(t, 0) for t in type_cnt.keys())
    else:
        raise ValueError(f"Unknown operator: {op!r}.")
    
    return lambda type_cnt: type_cnt == {QPLType(NUMBER): 1} or frozenset(type_cnt.keys()) in set(types_sets)
    

def rec_type_check(tree: QPLTree, schema: DBSchema, strict: bool) -> Tuple[Status, str]:
    for child in tree.children:
        status, error_message = rec_type_check(child, schema, strict)
        if status != Status.SUCCESS:
            return status, error_message
    try:
        root_type = qpl_tree_to_type(tree, schema, strict).type_count
        children_types = [qpl_tree_to_type(child, schema, strict).type_count for child in tree.children]
    except AssertionError as e:
        return Status.SYNTAX_ERROR, f"SyntaxError in line #{tree.line_num}: {str(e)}"
    except Exception as e:
        return Status.UNEXPECTED_ERROR, f"UnexpectedTypeInference error in line #{tree.line_num}: {str(e)}"

    if not children_types:
        return Status.SUCCESS, "Success."
    try:
        is_valid = check_and_resolve(tree.op, *children_types, strict=strict)
        if is_valid(root_type):
            return Status.SUCCESS, "Success."
        return Status.STRUCTURE_ERROR, f"StructureError in line #{tree.line_num}: Input types {' and '.join([types_str(child) for child in children_types])} are incompatible with output type {types_str(root_type)} for operator {tree.op.value!r}."
    except IncompatibleTypesError as e:
        return Status.OPERATOR_ERROR, f"OperatorError in line #{tree.line_num}: {str(e)}"
    

if __name__ == "__main__":
    import json
    from datasets import load_dataset

    STRICT = True
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

            status, error = rec_type_check(tree=tree, schema=schema, strict=STRICT)
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

