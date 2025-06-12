import re
import json
from collections import defaultdict
from typing import Dict, Any

from src.utils.chat_types import ChatTemplate, ChatMessage
from src.processors.qpl.base import QPLProcessor
from src.processors.base import ProcessorRegistry

from datasets import load_dataset


@ProcessorRegistry.register
class QPLCompleterProcessor(QPLProcessor):
    dataset_id = "d4nieldev/qpl-completer-ds"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        q_to_id = {}
        for id, content in self._db_content.items():
            question = content["question"]
            q_to_id[question] = id
        
        self.__q_to_id = q_to_id
        dataset = load_dataset(self.dataset_id)
        question_to_examples = defaultdict(list)
        for split in dataset:
            for example in dataset[split]:
                question_to_examples[example['question']].append(example)
        self.__sub_q_to_parents = defaultdict(list)
        for split in dataset:
            for example in dataset[split]:
                parent_question = example['parent_question']
                question = example['question']
                if parent_question is None:
                    continue
                for ex in question_to_examples[parent_question]:
                    if ex in self.__sub_q_to_parents[question]:
                        continue
                    self.__sub_q_to_parents[question].append(ex)

    def to_chat_template(self, example) -> ChatTemplate:
        db_id = example['db_id']

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
            + "<top> ::= Top [ <input> ] Rows [ <number> ] <output-non-qualif>\n"
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
        if line_num is None:
            line_num = example['qpl_line'].split('=')[0].strip()[1:]
        
        line_start = f"#{line_num} = {example['op']} "

        user = (
            f"Database Name: {db_id}\n\n"

            + "Database Schema:\n"
            + f"```DDL\n{self._create_table_prompt(example, log_when_parent_not_found=self.with_assistant)}```\n\n"

            + f"Question: {example['question'].strip()}\n\n"

            + "The QPL query that satisfies the question's intent is:\n\n"

            + f"```QPL\n{prefix_qpl_str}{line_start}"
        )

        if self.with_assistant:
            response = f"{example['qpl_line'].replace(line_start, '')}\n```"
            return ChatTemplate(
                messages=[
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                    ChatMessage(role="assistant", content=response),
                ]
            )
        else:
            return ChatTemplate(
                messages=[
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                ]
            )
    
    def _example_to_id(self, example: Dict[str, Any]) -> str:
        # get id
        if self.with_assistant:
            id = self.__q_to_id.get(example['question'])
            if id is None:
                # return id of parent
                potential_parents = self.__sub_q_to_parents.get(example['question'], [])
                ids = set()
                for parent in {frozenset(p.items()) for p in potential_parents}:
                    try:
                        ids.add(self._example_to_id(dict(parent)))
                    except ValueError:
                        continue
                if ids:
                    return ids.pop()
                # parent not found
                raise ValueError(f"Parent not found for question: {example['question']}")
            return id
        raise ValueError("Cannot get id in test mode")
