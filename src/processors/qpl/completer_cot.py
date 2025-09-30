import re
from datasets import load_dataset

from src.utils.chat_types import ChatML, Message
from src.processors.qpl.base import QPLProcessor
from src.processors.base import processorRegistry
from src.utils.tree import QPLTree


@processorRegistry.register
class QPLCompleterCotProcessor(QPLProcessor):
    dataset_id = "d4nieldev/qpl-completer-cot-ds"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self, subset: str = "balanced"):
        return load_dataset(self.dataset_id, subset)

    def to_chat_template(self, example, *, noise: float = 1.0, **kwargs) -> ChatML:
        system = (
            "Given a database schema, a QPL query prefix, and a natural language question, "
            + "complete the final line of the query so it completes the user request.\n\n"

            + "QPL is a formalism used to describe data retrieval operations over an SQL schema in a modular manner.\n"
            + "A QPL plan is a sequence of instructions for querying tabular data to answer a natural language question.\n\n"
            
            + "Below is the formal specification for each operation in valid QPL:\n"

            + "<qpl> ::= <line>+\n"
            + "<line> ::= #<integer> = <operator>\n"
            + "<operator> ::= <scan> | <aggregate> | <filter> | <sort> | <topsort> | <join> | <except> | <intersect> | <union>\n\n"

            + "-- Leaf operator\n"
            + "<scan> ::= Scan Table [ <table-name> ] <pred>? <distinct>? <output-non-qualif>\n\n"
            
            + "-- Unary operators\n"
            + "<aggregate> ::= Aggregate [ <input> ] <group-by>? <output-agg>\n"
            + "<filter> ::= Filter [ <input> ] <pred> <distinct>? <output-non-qualif>\n"
            + "<sort> ::= Sort [ <input> ] <order-by> <output-non-qualif>\n"
            + "<topsort> ::= TopSort [ <input> ] Rows [ <number> ] <order-by> <withTies>? <output-non-qualif>\n\n"
            
            + "-- Binary operators\n"
            + "<join> ::= Join [ <input> , <input> ] <pred>? <distinct>? <output-qualif>\n"
            + "<except> ::= Except [ <input> , <input> ] <pred> <output-qualif>\n"
            + "<intersect> ::= Intersect [ <input> , <input> ] <pred> <output-qualif>\n"
            + "<union> ::= Union [ <input> , <input> ] <output-qualif>\n\n"

            + "<group-by> ::= GroupBy [ <column-name> (, <column-name>)* ]\n"
            + "<order-by> ::= OrderBy [ <column-name> <direction> (, <column-name> <direction>)* ]\n"
            + "<withTies> ::= WithTies [ true | false ]\n"
            + "<direction> ::= ASC | DESC\n"
            + "<pred> ::= Predicate [ <comparison> (AND | OR <comparison)* ]\n"
            + "<distinct> ::= Distinct [ true | false ]\n"
            + "<output-non-qualif> ::= Output [ <column-name> (, <column-name>)* ]\n"
            + "<output-agg> ::= Output [ <agg-column-name> (, <agg-column-name>)* ]\n"
            + "<output-qualif> ::= Output [ <qualif-column-name> (, <qualif-column-name>)* ]\n"
            + "<agg-column-name> ::= countstar | <agg-func>(<column-name>) | <agg-func>(DISTINCT <column-name>)\n"
            + "<agg-func> ::= COUNT | SUM | AVG | MIN | MAX\n"
            + "<qualif-column-name> ::= #<number>.<column-name>\n\n"

            + "Using valid QPL, complete the last step in order to answer the question for the database schema provided below.\n\n"

            + "Before providing the final answer, you must first reason step by step about the question and the database schema."
        )

        prefix_qpl_str = example['prefix_qpl'].replace(' Top ', ' TopSort ')
        qpl_line = example.get('qpl_line', '').replace(' Top ', ' TopSort ')

        line_num = example.get('line_num', None)
        children_str = example.get('children_str', None)
        if line_num is None or children_str is None:
            if not 'qpl_line' in example:
                raise ValueError("Example must contain 'qpl_line' or 'line_num' and 'children_str'")
            line_num = qpl_line.split('=')[0].strip()[1:]
            if example['op'] == "Scan":
                children_str = "Table"
            else:
                m = re.match(
                    r"#(?P<idx>\d+) = (?P<op>\w+) \[ (?P<ins>[^\]]+) \] ((?P<opt>\w+) \[ (?P<arg>[^\]]+) \] )*Output \[ (?P<out>[^\]]+) \]",
                    qpl_line
                )
                if m:
                    children_str = f"[ {m.group('ins')} ]"
                else:
                    raise ValueError(f"QPL line does not match expected patterns: {qpl_line}")

        line_start = f"#{line_num} = {example['op'] if example['op'] != 'Top' else 'TopSort'} {children_str}"

        # inject noise to schema if needed
        if noise < 1.0:
            qpl_lines = [line.split(' ; ')[0] for line in prefix_qpl_str.split('\n') if line] + [qpl_line]
            table_cols = QPLTree.from_qpl_lines(qpl_lines).get_schema_items()
            schema_str = self._get_schema_str(db_id=example['db_id'], link_table_cols=table_cols, noise=noise)
        else:
            schema_str = self._get_schema_str(db_id=example['db_id'])

        user = (
            f"{schema_str}\n\n"

            + f"[Question]: {example['question'].strip()}\n\n"

            + f"[Prefix QPL]:\n"
            + f"```QPL\n{prefix_qpl_str}\n```\n\n"

            + f"[Complete]: `{line_start} ...`\n\n"

            + "First, understand the input stream, then determine operator-specific options, and finally select the output columns.\n"
            + "Provide your reasoning enclosed in <think> and </think> tags, and afterwards provide the final line of the QPL query in valid QPL.\n"
        )

        if self.with_assistant:
            response = f"<think>\n{example['cot']}\n</think>\n\n"
            response += f"```QPL\n{qpl_line}\n```"
            return ChatML(
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=user),
                    Message(role="assistant", content=response),
                ]
            )
        else:
            return ChatML(
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=user),
                ]
            )
