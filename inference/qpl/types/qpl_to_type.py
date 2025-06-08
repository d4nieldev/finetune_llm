import re
from typing import List, Dict, Tuple, ClassVar, Set, Union, Iterator
from enum import StrEnum
from dataclasses import dataclass

from utils.qpl.tree import QPLTree
from inference.qpl.types.schema_types import DBSchema, Table


class Operator(StrEnum):
    SCAN = "Scan"
    AGGREGATE = "Aggregate"
    FILTER = "Filter"
    TOP = "Top"
    SORT = "Sort"
    TOPSORT = "TopSort"
    JOIN = "Join"
    EXCEPT = "Except"
    INTERSECT = "Intersect"
    UNION = "Union"


@dataclass
class QPLType:
    table: Table
    """The table (entity) that this type represents"""

    aggregated: bool = False
    """Whether this type is aggregated or not"""


@dataclass
class QPLNodeOutput:
    cols: List[str]
    """The output columns of the QPL node"""

    aliases: List[str]
    """The aliases of the output columns of the QPL node - aliases[i] same as cols[i] if no alias is given"""

    types: List[QPLType]
    """The types that the output columns belong to - types[i] is the table of cols[i]"""

    COUNTSTAR: ClassVar[str] = 'countstar'


    def resolve_alias(self, alias: str) -> Tuple[str, QPLType]:
        # We start with a column name that is at least an alias
        if alias not in self.aliases:
            raise ValueError(f"Column {alias!r} not found.")
        
        # Find the column name and table in the child output
        col_idx = self.aliases.index(alias)
        colname = self.cols[col_idx]
        qpltype = self.types[col_idx]

        # Follow foreign keys to find the source table and column name
        colname, table = qpltype.table.src_colname_table(colname)
        
        return colname, QPLType(table=table, aggregated=qpltype.aggregated)


    @staticmethod
    def infer(captures_out: str, input: Union[Dict[int, "QPLNodeOutput"], Table]) -> "QPLNodeOutput":
        """
        Infers the output columns, aliases, and tables from the captures_out string and the children outputs.
        Args:
            captures_out (str): 
                The output columns string from the QPL captures.
            input (Dict[int, QPLNodeOutput] | Table): 
                In case of a unary / binary operation, this is a mapping of the children rows (row_number -> output).
                In the case of a scan operation, this is the table being scanned.
        Assumes:
            * The children outputs are already resolved meaning cols are the original column names in tables.
        Returns:
            QPLNodeOutput: An instance containing the inferred output columns, aliases, and tables.
        """

        cols = []
        aliases = []
        types = []

        for value in captures_out.split(","):
            split = re.split(r' AS ', value.strip(), flags=re.IGNORECASE)
            col_identifier = split[0].strip()  # could be alias in one of the children
            alias = split[1].strip() if len(split) > 1 else split[0].strip()

            if col_identifier == QPLNodeOutput.COUNTSTAR:
                if isinstance(input, Table):
                    raise ValueError(f"{QPLNodeOutput.COUNTSTAR!r} cannot be used with a single table output.")
                # countstar appears only in Aggregate operations 
                # while the Count_Star alias can appear higher in the tree
                colname = col_identifier
                child_tables = [qpltype.table for qpltype in list(input.values())[0].types]
                # take the most common table as a heuristic to be the table of the countstar
                # TODO: is there a better way to determine the table of countstar?
                table = max(set(child_tables), key=child_tables.count)
                qpltype = QPLType(table=table, aggregated=True)
            elif isinstance(input, Dict):
                qpltype = None

                # Find original column name and table
                if matches := re.findall(r'#(\d+)\.(\w+)', col_identifier):
                    # Binary operation - resolve to specified child input
                    row_idx, col_identifier = matches[0]
                    inp = input[int(row_idx)]
                    colname, qpltype = inp.resolve_alias(col_identifier)
                else:
                    # Unary operation

                    # Check if the column is wrapped with an aggregation function
                    match_agg = re.match(r'\s*(\w+)\s*\(\s*(\w+)\s*\)\s*', col_identifier)
                    aggregated = False
                    if match_agg:
                        func, col_identifier = match_agg.groups()
                        aggregated = True
                    
                    # Automatically resolve to a child input
                    inp = list(input.values())[0]
                    colname, qpltype = inp.resolve_alias(col_identifier)
                    qpltype.aggregated = aggregated
            else:
                # Scan operation - resolve to the table directly
                if col_identifier not in input.column_names:
                    raise ValueError(f"Column '{col_identifier}' not found in table {input.name}.")
                colname = col_identifier
                qpltype = QPLType(table=input, aggregated=False)
            
            cols.append(colname)
            aliases.append(alias)
            types.append(qpltype)
        
        return QPLNodeOutput(cols=cols, aliases=aliases, types=types)
    
    @property
    def type_set(self) -> Set[QPLType]:
        """Returns a set of unique QPLTypes in the output."""
        return set(self.types)
    
    def __iter__(self) -> Iterator[Tuple[str, str, QPLType]]:
        """Iterate over the output columns, aliases, and types."""
        return iter(zip(self.cols, self.aliases, self.types))


def scan_type(captures: Dict, schema: DBSchema) -> QPLNodeOutput:
    table_name = captures["table"]
    table = schema.tables.get(table_name)
    if not table:
        raise ValueError(f"Table {table_name} not found in schema.")

    return QPLNodeOutput.infer(captures['out'], table)

def aggregate_type(agg_node: QPLTree, captures: Dict, schema: DBSchema) -> QPLNodeOutput:
    ins = [int(x[1:]) for x in re.split(r"\s*, ", captures["ins"][0])]
    child_idx = ins[0]

    option_names = captures["opt"]
    args = captures["arg"]
    opts = dict(zip(option_names, args))

    child_output = qpl_tree_to_type(agg_node.children[0], schema)
    node_output = QPLNodeOutput.infer(captures['out'], {child_idx: child_output})

    # Verify that the output columns match the GroupBy columns
    gb_cols = [alias.strip() for alias in opts.get("GroupBy", "").split(",")]
    for col, alias, qpltype in node_output:
        if alias in gb_cols:
            idx = child_output.aliases.index(alias)
            child_aggregated = child_output.types[idx].aggregated
            if child_aggregated:
                # Interesting edge case
                # The column is the result of a previous aggregation and now used in GroupBy
                raise ValueError(f"Column {alias!r} is aggregated in the child output but used in GroupBy.")
            assert child_aggregated == qpltype.aggregated, f"Column {alias!r} in GroupBy and thus must not be aggregated."
        else:
            assert qpltype.aggregated, f"Column {alias!r} not in GroupBy and thus must be aggregated."
    
    # Verify that all GroupBy columns are present in the output
    for colname in gb_cols:
        if colname not in node_output.aliases:
            raise AssertionError(f"GroupBy column {colname!r} not found in aggregate output: {node_output.aliases}.")
    
    return node_output


def filter_type(filter_node: QPLTree, captures: Dict, schema: DBSchema) -> QPLNodeOutput:
    pass

def top_type(top_node: QPLTree, captures: Dict, schema: DBSchema) -> QPLNodeOutput:
    pass

def sort_type(sort_node: QPLTree, captures: Dict, schema: DBSchema) -> QPLNodeOutput:
    pass

def topsort_type(topsort_node: QPLTree, captures: Dict, schema: DBSchema) -> QPLNodeOutput:
    pass

def join_type(join_node: QPLTree, captures: Dict, schema: DBSchema) -> QPLNodeOutput:
    pass

def except_type(except_node: QPLTree, captures: Dict, schema: DBSchema) -> QPLNodeOutput:
    pass

def intersect_type(intersect_node: QPLTree, captures: Dict, schema: DBSchema) -> QPLNodeOutput:
    pass

def union_type(union_node: QPLTree, captures: Dict, schema: DBSchema) -> QPLNodeOutput:
    pass


def qpl_tree_to_type(qpl_tree: QPLTree, schema: DBSchema) -> QPLNodeOutput:
    scan_regex = re.compile(
        r"#(?P<idx>\d+) = Scan Table \[ (?P<table>\w+) \]( Predicate \[ (?P<pred>[^\]]+) \])?( Distinct \[ (?P<distinct>true) \])? Output \[ (?P<out>[^\]]+) \]"
    )
    flat_qpl_line_pattern = re.compile(
        r"#(?P<idx>\d+) = (?P<op>\w+) \[ (?P<ins>[^\]]+) \] ((?P<opt>\w+) \[ (?P<arg>[^\]]+) \] )*Output \[ (?P<out>[^\]]+) \]"
    )

    def rec(node: QPLTree) -> QPLNodeOutput:
        if m := scan_regex.match(node.qpl_line):
            return scan_type(m.groupdict(), schema)
        elif m := flat_qpl_line_pattern.match(node.qpl_line):
            captures = m.groupdict()
            op = captures["op"][0]
            ins = [int(x[1:]) for x in re.split(r"\s*, ", captures["ins"][0])]
            option_names = captures["opt"]
            args = captures["arg"]
            opts = dict(zip(option_names, args))
            output_list = re.split(r"\s*, ", captures["out"][0])

            if op == Operator.AGGREGATE:
                return aggregate_type(node, captures, schema)
            elif op == Operator.FILTER:
                return filter_type(node, captures, schema)
            elif op == Operator.TOP:
                return top_type(node, captures, schema)
            elif op == Operator.SORT:
                return sort_type(node, captures, schema)
            elif op == Operator.TOPSORT:
                return topsort_type(node, captures, schema)
            elif op == Operator.JOIN:
                return join_type(node, captures, schema)
            elif op == Operator.EXCEPT:
                return except_type(node, captures, schema)
            elif op == Operator.INTERSECT:
                return intersect_type(node, captures, schema)
            elif op == Operator.UNION:
                return union_type(node, captures, schema)

        raise ValueError(f"Could not infer type for: {node.qpl_line}")

    return rec(qpl_tree)

    
