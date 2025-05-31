import re
from typing import List, Dict, Optional, Tuple
from enum import Enum

from inference.qpl.types.qpl_types import *
from utils.qpl.tree import QPLTree
from inference.qpl.types.schema_types import DBSchema


class Operator(Enum):
    SCAN = "Scan"
    AGGREGATE = "Aggregate"
    FILTER = "Filter"
    SORT = "Sort"
    TOPSORT = "TopSort"
    JOIN = "Join"
    EXCEPT = "Except"
    INTERSECT = "Intersect"
    UNION = "Union"


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def scan_type(scan_node: QPLTree, schema: DBSchema) -> Optional[QPLType]:
    scan_regex = re.compile(
        r"#(?P<idx>\d+) = Scan Table \[ (?P<table>\w+) \]( Predicate \[ (?P<pred>[^\]]+) \])?( Distinct \[ (?P<distinct>true) \])? Output \[ (?P<out>[^\]]+) \]"
    )
    if m := scan_regex.match(scan_node.qpl_row):
        captures = m.groupdict()
        table_name = captures["table"]
        out_orig_cols = [col.split(" AS ")[0].strip() for col in captures["out"].split(",")]
        
        out_types = set()
        for col in out_orig_cols:
            if is_number(col):
                out_types.add(Number())
            elif (
                col in schema[table_name].columns and
                col not in [fk.from_col for fk in schema[table_name].fks] and
                col not in [pk.col_name for pk in schema[table_name].pks]
            ):
                # col is a non-key column
                out_types.add(NoPK(Entity(table_name)))
            elif col not in schema[table_name].columns:
                raise ValueError(f"Column {col} not found in table {table_name}")
        
        for table in schema.tables.values():
            if set(schema[table_name].pks).issubset(out_orig_cols):
                out_types.add(Entity(table.name))

def aggregate_type(agg_node: QPLTree, schema: DBSchema) -> Optional[QPLType]:
    pass

def filter_type(filter_node: QPLTree, schema: DBSchema) -> Optional[QPLType]:
    pass

def sort_type(sort_node: QPLTree, schema: DBSchema) -> Optional[QPLType]:
    pass

def topsort_type(topsort_node: QPLTree, schema: DBSchema) -> Optional[QPLType]:
    pass

def join_type(join_node: QPLTree, schema: DBSchema) -> Optional[QPLType]:
    pass

def except_type(except_node: QPLTree, schema: DBSchema) -> Optional[QPLType]:
    pass

def intersect_type(intersect_node: QPLTree, schema: DBSchema) -> Optional[QPLType]:
    pass

def union_type(union_node: QPLTree, schema: DBSchema) -> Optional[QPLType]:
    pass


def qpl_to_type(qpl_lines: List[str], schema: DBSchema) -> QPLType:
    root = QPLTree.from_qpl_lines(qpl_lines)

    def rec(node: QPLTree) -> QPLType:
        if t := scan_type(node, schema):
            return t
        elif t := aggregate_type(node, schema):
            return t
        elif t := filter_type(node, schema):
            return t
        elif t := sort_type(node, schema):
            return t
        elif t := topsort_type(node, schema):
            return t
        elif t := join_type(node, schema):
            return t
        elif t := except_type(node, schema):
            return t
        elif t := intersect_type(node, schema):
            return t
        elif t := union_type(node, schema):
            return t
        else:
            raise ValueError(f"Could not infer type for: {node.qpl_row}")

    return rec(root)
