import re
import json
from collections import defaultdict
from typing import Dict, Any, Literal

from src.utils.chat_types import ChatML, Message
from src.processors.qpl.base import QPLProcessor
from src.processors.base import processorRegistry


@processorRegistry.register
class QPLCompleterProcessor(QPLProcessor):
    dataset_id = "d4nieldev/qpl-completer-ds"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_chat_template(self, example) -> ChatML:
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
            + "<aggregate> ::= Aggregate [ <input> ] <group-by>? <output-non-qualif>\n"
            + "<filter> ::= Filter [ <input> ] <pred> <distinct>? <output-non-qualif>\n"
            + "<sort> ::= Sort [ <input> ] <order-by> <withTie>? <output-non-qualif>\n"
            + "<topsort> ::= TopSort [ <input> ] Rows [ <number> ] <order-by> <withTies>? <output-non-qualif>\n\n"
            
            + "-- Binary operators\n"
            + "<join> ::= Join [ <input> , <input> ] <pred>? <distinct>? <output-qualif>\n"
            + "<except> ::= Except [ <input> , <input> ] <pred> <output-qualif>\n"
            + "<intersect> ::= Intersect [ <input> , <input> ] <pred>? <output-qualif>\n"
            + "<union> ::= Union [ <input> , <input> ] <output-qualif>\n\n"

            + "<group-by> ::= GroupBy [ <column-name> (, <column-name>)* ]\n"
            + "<order-by> ::= OrderBy [ <column-name> <direction> (, <column-name> <direction>)* ]\n"
            + "<withTies> ::= WithTies [ true | false ]\n"
            + "<direction> ::= ASC | DESC\n"
            + "<pred> ::= Predicate [ <comparison> (AND | OR <comparison)* ]\n"
            + "<distinct> ::= Distinct [ true | false ]\n"
            + "<output-non-qualif> ::= Output [ <column-name> (, <column-name>)* ]\n"
            + "<output-qualif> ::= Output [ <qualif-column-name> (, <qualif-column-name>)* ]\n"
            + "<qualif-column-name> ::= # <number> . <column-name>"
        )

        prefix_qpl_str = ' ;\n'.join(example['prefix_qpl'].split(' ; '))
        if example['prefix_qpl'] != "":
            prefix_qpl_str += " ;\n"

        line_num = example.get('line_num', None)
        children_str = example.get('children_str', None)
        if line_num is None or children_str is None:
            if not 'qpl_line' in example:
                raise ValueError("Example must contain 'qpl_line' or 'line_num' and 'children_str'")
            line_num = example['qpl_line'].split('=')[0].strip()[1:]
            if example['op'] == "Scan":
                children_str = "Table"
            else:
                m = re.match(
                    r"#(?P<idx>\d+) = (?P<op>\w+) \[ (?P<ins>[^\]]+) \] ((?P<opt>\w+) \[ (?P<arg>[^\]]+) \] )*Output \[ (?P<out>[^\]]+) \]",
                    example['qpl_line']
                )
                if m:
                    children_str = f"[ {m.group('ins')} ]"
                else:
                    raise ValueError(f"QPL line does not match expected patterns: {example['qpl_line']}")

        line_start = f"#{line_num} = {example['op']} {children_str} "

        user = (
            f"{self._get_schema_str(example['db_id'])}\n\n"

            + f"Question: {example['question'].strip()}\n\n"

            + "The QPL query that satisfies the question's intent is:\n\n"

            + f"```QPL\n{prefix_qpl_str}{line_start}"
        )

        if self.with_assistant:
            response = f"{example['qpl_line'].replace(line_start, '')}\n```"
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
