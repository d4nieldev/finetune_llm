from collections import defaultdict

from src.utils.chat_types import ChatTemplate, ChatMessage
from src.prompters.qpl.decomposer import QPLDecomposerPrompter
from src.prompters.base import PrompterRegistry

from datasets import load_dataset


@PrompterRegistry.register
class QPLDecomposerCotPrompter(QPLDecomposerPrompter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def dataset_id(self) -> str:
        return "d4nieldev/qpl-decomposer-cot-ds"

    def load_dataset(self):
        load_dataset(self.dataset_id, 'balanced')

    def to_chat_template(self, example) -> ChatTemplate:
        db_id = example['db_id']

        system = (
            "Given a database schema and a question in natural language, "
            + "you must predict the toplevel QPL operator and if needed, decompose the input question into one or two "
            + "simpler sub-questions which describe the arguments of the toplevel operator.\n\n"

            + "The toplevel QPL operators are:\n"
            + "**Scan** - Scan all rows in a table with optional filtering predicate (no decomposition needed - the question is atomic)\n"
            + "**Aggregate** - Aggregate a stream of tuples, optionally using a grouping criterion into a stream of groups (1 sub-question)\n"
            + "**Filter** - Remove tuples from a stream that do not match a predicate (1 sub-question)\n"
            + "**Sort** - Sort a stream according to a sorting expression (1 sub-question)\n"
            + "**TopSort** - Select the top-K tuples from a stream according to a sorting expression (1 sub-question)\n"
            + "**Join** - Perform a logical join operation between two streams based on a join condition (2 sub-questions)\n"
            + "**Except** - Compute the set difference between two streams of tuples (2 sub-questions)\n"
            + "**Intersect** - Compute the set intersection between two streams of tuples (2 sub-questions)\n"
            + "**Union** - Compute the set union between two streams of tuples (2 sub-questions)\n\n"

            + "Before providing the final answer, you must first reason step by step about the question and the database schema.\n"
            + "First, determine the operator, then formulate the sub-questions if needed, and finally justify the decomposition."
        )

        user = (
            f"Database Name: {db_id}\n\n"

            + "Database Schema:\n"
            + f"```DDL\n{self._create_table_prompt(example, log_when_parent_not_found=self.with_assistant)}\n```\n\n"

            + f"""Question: {example["question"].strip()}\n\n"""

            + "Provide your reasoning enclosed in <think> and </think> tags, and afterwards provide the final answer in the following format:\n"
            + "The first line of the final answer should be the toplevel operator, the following lines should be the predicted sub-questions."
        )

        if self.with_assistant:
            response = f"<think>\n{example['cot']}\n</think>\n\n"
            op = example['op']
            if op == "Top":
                # Special case (only 1 row)
                op = "TopSort"
            response += op
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
