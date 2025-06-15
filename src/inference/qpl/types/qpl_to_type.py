from typing import List, Dict, Tuple, ClassVar, Union, Iterator, Any, Literal
from collections import Counter
from dataclasses import dataclass
import logging as log

import regex as re

from src.utils.qpl.tree import QPLTree, Operator
from src.inference.qpl.types.schema_types import DBSchema, Table


NUMBER = "Number"

class QPLType:
    def __init__(self, entity: Union[Table, Literal['Number']], aggregated: bool = False):
        self.entity: Union[Table, Literal['Number']] = entity
        self.aggregated = aggregated

    @property
    def aggregated(self) -> bool:
        """Returns whether this type is aggregated or not."""
        return self._aggregated
    
    @aggregated.setter
    def aggregated(self, value: bool):
        """Sets whether this type is aggregated or not."""
        if self.entity == NUMBER:
            self._aggregated = True
        else:
            self._aggregated = value

    def __str__(self) -> str:
        if self.entity == NUMBER:
            return NUMBER
        if self.aggregated:
            return f"Aggregated[{self.entity.name}]"
        return f"{self.entity.name}"
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, QPLType):
            return False
        return str(self) == str(other)
    
    def __hash__(self) -> int:
        return hash(str(self))


def types_str(type_count: Dict[QPLType, int]) -> str:
    """
    Converts a dictionary of QPLType to their counts into a string representation.
    """
    return ", ".join(f"{t}({count})" for t, count in type_count.items())


@dataclass
class QPLNodeOutput:
    cols: List[str]
    """The output columns of the QPL node"""

    aliases: List[str]
    """The aliases of the output columns of the QPL node - aliases[i] same as cols[i] if no alias is given"""

    types: List[QPLType]
    """The types that the output columns belong to - types[i] is the table of cols[i]"""

    COUNTSTAR: ClassVar[str] = 'countstar'
    """Special case for COUNT(*) aggregation function"""

    ONE: ClassVar[str] = '1'
    """Special case for a column value of 1, used in some QPL operations"""

    AGG_NUM: ClassVar[List[str]] = ['count', 'sum', 'avg']
    """List of aggregation functions that return a numeric type"""

    AGG_COL: ClassVar[List[str]] = ['max', 'min']
    """List of aggregation functions that return a column type"""


    def resolve_alias(self, alias: str) -> Tuple[str, QPLType]:
        # We start with a column name that is at least an alias
        if alias not in self.aliases:
            raise ValueError(f"Column {alias!r} not found.")
        
        # Find the column name and table in the child output
        col_idx = self.aliases.index(alias)
        colname = self.cols[col_idx]
        qpltype = self.types[col_idx]

        # Follow foreign keys to find the source table and column name
        table = NUMBER
        if isinstance(qpltype.entity, Table):
            colname, table = qpltype.entity.src_colname_table(colname)
        
        return colname, QPLType(entity=table, aggregated=qpltype.aggregated)


    @staticmethod
    def infer(captures_out: str, inputs: Union[Dict[int, "QPLNodeOutput"], Table]) -> "QPLNodeOutput":
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
            value = value.lower()
            as_split = re.split(r' AS ', value.strip(), flags=re.IGNORECASE)
            col_identifier = as_split[0].strip()  # could be alias in one of the children
            alias = as_split[1].strip() if len(as_split) > 1 else as_split[0].split('.')[-1].strip()

            if col_identifier in [QPLNodeOutput.COUNTSTAR, QPLNodeOutput.ONE]:
                # Special cases
                if col_identifier == QPLNodeOutput.COUNTSTAR and isinstance(inputs, Table):
                    raise ValueError(f"{QPLNodeOutput.COUNTSTAR!r} is an aggregation function, hence should be used in Aggregate.")
                colname = col_identifier
                qpltype = QPLType(entity=NUMBER, aggregated=True)
            elif isinstance(inputs, Dict):
                qpltype = None

                # Find original column name and table
                if len(inputs) == 2:
                    # Binary operation - resolve to specified child input
                    if not (matches := re.findall(r'[#(\d+)\.]?(\w+)', col_identifier)):
                        raise ValueError(f"Unexpected column identifier for binary operation: {col_identifier!r}.")

                    if len(matches) == 2:
                        row_idx, col_identifier = matches
                        inp = inputs[int(row_idx)]
                        colname, qpltype = inp.resolve_alias(col_identifier)
                    else:
                        col_identifier = matches[0]
                        found = False
                        for inp in inputs.values():
                            if col_identifier in inp.aliases:
                                colname, qpltype = inp.resolve_alias(col_identifier)
                                found = True
                                break
                        if not found:
                            raise ValueError(f"Column {col_identifier!r} not found in any child input: {inputs.keys()}.")
                else:
                    # Unary operation

                    # Check if the column is wrapped with an aggregation function
                    col_identifier = col_identifier.replace('distinct', '').strip()
                    match_agg = re.match(r'\s*(\w+)\s*\(\s*(\w+)\s*\)\s*', col_identifier)
                    aggregated = False
                    if match_agg:
                        func, col_identifier = match_agg.groups()
                        aggregated = True
                        if func not in QPLNodeOutput.AGG_NUM + QPLNodeOutput.AGG_COL:
                            raise ValueError(f"Unknown aggregation function {func!r} in {col_identifier!r}.")
                        if func in QPLNodeOutput.AGG_NUM:
                            colname = col_identifier
                            qpltype = QPLType(entity=NUMBER, aggregated=True)
                    if not match_agg or func in QPLNodeOutput.AGG_COL:
                        # Automatically resolve to a child input
                        inp = list(inputs.values())[0]
                        colname, qpltype = inp.resolve_alias(col_identifier)
                        qpltype.aggregated = qpltype.aggregated or aggregated
            else:
                # Scan operation - resolve to the table directly
                if col_identifier not in inputs.column_names:
                    raise ValueError(f"Column '{col_identifier}' not found in table {inputs.name}.")
                colname, table = inputs.src_colname_table(col_identifier)
                qpltype = QPLType(entity=table, aggregated=False)
            
            cols.append(colname)
            aliases.append(alias)
            types.append(qpltype)
        
        return QPLNodeOutput(cols=cols, aliases=aliases, types=types)
    
    @property
    def type_count(self) -> Dict[QPLType, int]:
        """Returns a dictionary of types and their counts."""
        return dict(Counter(self.types))
    
    def __iter__(self) -> Iterator[Tuple[str, str, QPLType]]:
        """Iterate over the output columns, aliases, and types."""
        return iter(zip(self.cols, self.aliases, self.types))
    
    def to_json(self) -> Dict[str, Any]:
        return {
            "outputs": [
                {
                    "col": col,
                    "alias": alias,
                    "col_type": str(qpltype),
                }
                for col, alias, qpltype in self
            ],
            "type": types_str(self.type_count),
        }


def scan_type(groups: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    table_name = groups["table"]
    table = schema.get_table(table_name)
    if not table:
        raise ValueError(f"Table {table_name} not found in schema.")

    return QPLNodeOutput.infer(groups['out'], table)

def aggregate_type(agg_node: QPLTree, captures: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    ins = [int(x[1:]) for x in re.split(r"\s*, ", captures["ins"][0])]
    child_idx = ins[0]

    option_names = captures["opt"]
    args = captures["arg"]
    opts = dict(zip(option_names, args))

    child_output = qpl_tree_to_type(agg_node.children[0], schema)
    node_output = QPLNodeOutput.infer(captures['out'][0], {child_idx: child_output})

    if strict:
        # Verify that the output columns match the GroupBy columns
        gb_cols = [alias.lower().strip() for alias in opts.get("GroupBy", "").split(",") if "GroupBy" in opts]
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


def same_child_type_change_cols(node: QPLTree, captures: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    ins = [int(x[1:]) for x in re.split(r"\s*, ", captures["ins"][0])]

    inputs = {
        child_idx: qpl_tree_to_type(node.children[i], schema)
        for i, child_idx in enumerate(ins)
    }

    return QPLNodeOutput.infer(captures['out'][0], inputs)

def filter_type(filter_node: QPLTree, captures: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    return same_child_type_change_cols(filter_node, captures, schema, strict)


def top_type(top_node: QPLTree, captures: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    return same_child_type_change_cols(top_node, captures, schema, strict)


def sort_type(sort_node: QPLTree, captures: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    return same_child_type_change_cols(sort_node, captures, schema, strict)


def topsort_type(topsort_node: QPLTree, captures: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    return same_child_type_change_cols(topsort_node, captures, schema, strict)


def join_type(join_node: QPLTree, captures: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    ins = [int(x[1:]) for x in re.split(r"\s*, ", captures["ins"][0])]
    lhs, rhs = ins

    lhs_output = qpl_tree_to_type(join_node.children[0], schema)
    rhs_output = qpl_tree_to_type(join_node.children[1], schema)

    return QPLNodeOutput.infer(captures_out=captures['out'][0], inputs={lhs: lhs_output, rhs: rhs_output})


def except_type(except_node: QPLTree, captures: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    return same_child_type_change_cols(except_node, captures, schema, strict)


def intersect_type(intersect_node: QPLTree, captures: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    return same_child_type_change_cols(intersect_node, captures, schema, strict)


def union_type(union_node: QPLTree, captures: Dict, schema: DBSchema, strict: bool) -> QPLNodeOutput:
    return same_child_type_change_cols(union_node, captures, schema, strict)


def qpl_tree_to_type(qpl_tree: QPLTree, schema: DBSchema, strict: bool = True) -> QPLNodeOutput:
    scan_regex = re.compile(
        r"#(?P<idx>\d+) = Scan Table \[ (?P<table>\w+) \]( Predicate \[ (?P<pred>[^\]]+) \])?( Distinct \[ (?P<distinct>true) \])? Output \[ (?P<out>[^\]]+) \]"
    )
    flat_qpl_line_pattern = re.compile(
        r"#(?P<idx>\d+) = (?P<op>\w+) \[ (?P<ins>[^\]]+) \] ((?P<opt>\w+) \[ (?P<arg>[^\]]+) \] )*Output \[ (?P<out>[^\]]+) \]"
    )

    def rec(node: QPLTree) -> QPLNodeOutput:
        if m := scan_regex.match(node.qpl_line):
            return scan_type(m.groupdict(), schema, strict)
        elif m := flat_qpl_line_pattern.match(node.qpl_line):
            captures = m.capturesdict()
            op = captures["op"][0]

            if op == Operator.AGGREGATE:
                return aggregate_type(node, captures, schema, strict)
            elif op == Operator.FILTER:
                return filter_type(node, captures, schema, strict)
            elif op == Operator.TOP:
                return top_type(node, captures, schema, strict)
            elif op == Operator.SORT:
                return sort_type(node, captures, schema, strict)
            elif op == Operator.TOPSORT:
                return topsort_type(node, captures, schema, strict)
            elif op == Operator.JOIN:
                return join_type(node, captures, schema, strict)
            elif op == Operator.EXCEPT:
                return except_type(node, captures, schema, strict)
            elif op == Operator.INTERSECT:
                return intersect_type(node, captures, schema, strict)
            elif op == Operator.UNION:
                return union_type(node, captures, schema, strict)

        raise ValueError(f"Could not infer type for: {node.qpl_line}")

    return rec(qpl_tree)


if __name__ == "__main__":
    import json
    from datasets import load_dataset
    import src.utils.qpl.paths as p

    DB_ID = 'battle_death'
    SPLIT = 'validation'

    dataset = load_dataset("d4nieldev/qpl-completer-ds", split=SPLIT)
    dataset = map(lambda row: {
        'db_id': row['db_id'],
        'question': row['question'],
        'qpl': [line.split(";")[0].strip() for line in row['prefix_qpl'].split("\n") if line] + [row['qpl_line']]
    }, dataset)
    dataset = filter(lambda row: row['db_id'] == DB_ID, dataset)
    dataset = list(dataset)

    qpl_trees = [QPLTree.from_qpl_lines(row['qpl']) for row in dataset]
    db_schemas = DBSchema.from_db_schemas_file(p.DB_SCHEMAS_JSON_PATH)

    types = []
    errs = 0
    for row, qpl_tree in zip(dataset, qpl_trees):
        db_schema = db_schemas[row['db_id']]
        try:
            types.append(qpl_tree_to_type(qpl_tree, db_schema).to_json())
        except Exception as e:
            types.append({"error": str(e)})
            log.warning(f"Error processing QPL tree for QPL: {' ; '.join(row['qpl'])}\n{e}")
            errs += 1

    print(f"Error rate: {errs}/{len(types)} ({errs/len(types)*100:.2f}%)")
    
    output = []
    for row, qpl_type in zip(dataset, types):
        output.append(row | qpl_type)

    with open(f"qpl_types_{DB_ID}.json", "w") as f:
        json.dump(output, f, indent=2)

