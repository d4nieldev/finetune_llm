from typing import Dict, Any, List

from src.utils.chat_types import ChatTemplate, ChatMessage
from src.prompters.qpl.base import QPLPrompter
from src.prompters.base import PrompterRegistry


@PrompterRegistry.register
class QPLDecomposerPrompter(QPLPrompter):
    dataset_id = "d4nieldev/qpl-decomposer-ds"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_chat_template(self, example) -> ChatTemplate:
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
            + "**Top** - Select the top-K tuples from a stream (1 sub-question)\n"
            + "**TopSort** - Select the top-K tuples from a stream according to a sorting expression (1 sub-question)\n"
            + "**Join** - Perform a logical join operation between two streams based on a join condition (2 sub-questions)\n"
            + "**Except** - Compute the set difference between two streams of tuples (2 sub-questions)\n"
            + "**Intersect** - Compute the set intersection between two streams of tuples (2 sub-questions)\n"
            + "**Union** - Compute the set union between two streams of tuples (2 sub-questions)\n\n"
        )

        user = (
            f"{self._get_schema_str(db_id)}\n\n"

            + f"""Question: {example["question"].strip()}\n\n"""

            + "The first line of the output should be the toplevel operator, the following lines should be the predicted sub-questions."
        )

        if self.with_assistant:
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
