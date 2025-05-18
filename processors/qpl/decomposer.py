from typing import Dict, Any

from custom_types import ChatTemplate, ChatMessage
from processors.qpl.base import QPLProcessor
from processors.base import ProcessorRegistry

from datasets import load_dataset


@ProcessorRegistry.register
class QPLDecomposerProcessor(QPLProcessor):
    dataset_id = "bgunlp/question_decomposer_ds"

    def __init__(self):
        super().__init__()
        
        q_to_id = {}
        for id, content in self._db_content.items():
            question = content["question"]
            q_to_id[question] = id
        
        self.__q_to_id = q_to_id
        self.__dataset = load_dataset(self.dataset_id)

    def to_chat_template(self, example, assistant_response: bool = False) -> ChatTemplate:
        db_id = example['db_id']

        system = (
            "Given a database schema and a question in natural language, "
            + "you must predict the toplevel operator and if needed, decompose the input question into one or two "
            + "simpler sub-questions which describe the arguments of the toplevel operator.\n\n"

            + "The toplevel operators are:\n"
            + "**Scan** - Scan all rows in a table with optional filtering predicate (no decomposition needed - the question is atomic)\n"
            + "**Aggregate** - Aggregate a stream of tuples using a grouping criterion into a stream of groups (1 sub-question)\n"
            + "**Filter** - Remove tuples from a stream that do not match a predicate (1 sub-question)\n"
            + "**Sort** - Sort a stream according to a sorting expression (1 sub-question)\n"
            + "**TopSort** - Select the top-K tuples from a stream according to a sorting expression (1 sub-question)\n"
            + "**Join** - Perform a logical join operation between two streams based on a join condition (2 sub-questions)\n"
            + "**Except** - Compute the set difference between two streams of tuples (2 sub-questions)\n"
            + "**Intersect** - Compute the set intersection between two streams of tuples (2 sub-questions)\n"
            + "**Union** - Compute the set union between two streams of tuples (2 sub-questions)\n\n"
        )

        user = (
            f"Database Name: {db_id}\n\n"

            + "Database Schema:\n"
            + f"```DDL\n{self._create_table_prompt(example, log_when_parent_not_found=assistant_response)}\n```\n\n"

            + f"""Question: {example["question"].strip()}\n\n"""

            + "The first line of the output should be the toplevel operator, the following lines should be the predicted sub-questions."
        )

        if assistant_response:
            response = f"{example['op']}"
            if example['sub_question_1']:
                response += f"\n{example['sub_question_1']}"
            if example['sub_question_2']:
                response += f"\n{example['sub_question_2']}"

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
        id = self.__q_to_id.get(example['question'])
        if id is None:
            # return id of parent
            for dataset in self.__dataset.values():  # type: ignore
                for ex in dataset:
                    if ex['sub_question_1'] == example['question'] or ex['sub_question_2'] == example['question']:
                        return self._example_to_id(ex)
            # parent not found
            raise ValueError(f"Parent not found for question: {example['question']}")
        return id
