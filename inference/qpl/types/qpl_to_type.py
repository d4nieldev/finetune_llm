import re
from typing import List, Dict, Optional
from enum import Enum

from inference.qpl.types.types import *
from utils.qpl.tree import QPLTree, get_qpl_tree



@dataclass
class ForeignKey:
    columns: List[str]
    to: List["Table"]

    def __post_init__(self):
        assert (len(self.columns) == len(self.to)), "Foreign key columns and referenced tables must have the same length"


@dataclass
class Table:
    name: str
    columns: List[str]
    pks: List[str]
    fks: List[ForeignKey]

    def __post_init__(self):
        assert all(pk in self.columns for pk in self.pks), f"Primary keys {self.pks} must be a subset of columns {self.columns}"
        assert all(col in self.columns for fk in self.fks for col in fk.columns), f"Foreign keys {self.fks} must be a subset of columns {self.columns}"


class Schema:
    __tables: Dict[str, Table]

    def __post_init__(self):
        assert len(self.__tables) == len(set(table.name for table in self.__tables.values())), "Table names must be unique"
        assert all(fk.to in self.__tables for table in self.__tables.values() for fk in table.fks), "Foreign keys must reference existing tables"

    @property
    def entities(self) -> List[Entity]:
        # primary keys that are not foreign keys to other tables distinguish the table as an entity
        return [
            Entity(table.name)
            for table in self.__tables.values()
            if set(table.pks).difference([fk_id for fk in table.fks for fk_id in fk.columns])
        ]
    
    @property
    def tables(self) -> List[Table]:
        return list(self.__tables.values())
    
    def __getitem__(self, item: str) -> Table:
        return self.__tables[item]


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


def scan_type(scan_node: QPLTree, schema: Schema) -> Optional[QPLType]:
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
                col not in [fk_col for fk in schema[table_name].fks for fk_col in fk.columns] and
                col not in schema[table_name].pks
            ):
                # col is a non-key column
                out_types.add(Partial(Entity(table_name)))
            elif col not in schema[table_name].columns:
                raise ValueError(f"Column {col} not found in table {table_name}")
        
        for table in schema.tables:
            if set(schema[table_name].pks).issubset(out_orig_cols):
                out_types.add(Entity(table.name))

def aggregate_type(agg_node: QPLTree, schema: Schema) -> Optional[QPLType]:
    pass

def filter_type(filter_node: QPLTree, schema: Schema) -> Optional[QPLType]:
    pass

def sort_type(sort_node: QPLTree, schema: Schema) -> Optional[QPLType]:
    pass

def topsort_type(topsort_node: QPLTree, schema: Schema) -> Optional[QPLType]:
    pass

def join_type(join_node: QPLTree, schema: Schema) -> Optional[QPLType]:
    pass

def except_type(except_node: QPLTree, schema: Schema) -> Optional[QPLType]:
    pass

def intersect_type(intersect_node: QPLTree, schema: Schema) -> Optional[QPLType]:
    pass

def union_type(union_node: QPLTree, schema: Schema) -> Optional[QPLType]:
    pass


def get_type(qpl_lines: List[str], schema: Schema) -> QPLType:
    root = get_qpl_tree(qpl_lines)

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
