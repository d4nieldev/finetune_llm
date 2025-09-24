from datasets import DatasetDict

from src.utils.chat_types import ChatML, Message
from src.prompters.qpl.base import QPLPrompter
from src.prompters.base import PrompterRegistry

from datasets import load_dataset


@PrompterRegistry.register
class QPLDecomposerCotPrompter(QPLPrompter):
    dataset_id = "d4nieldev/qpl-decomposer-cot-ds"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self, subset: str = "balanced"):
        return load_dataset(self.dataset_id, subset)

    @property
    def system_prompt(self) -> str:
        return (
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

            + "Before providing the final answer, you must first reason step by step about the question and the database schema."
        )
    
    def user_prompt(self, db_id: str, question: str) -> str:
        return (
            f"{self._get_schema_str(db_id)}\n\n"

            + f"""[Question]: {question.strip()}\n\n"""

            + "First, determine the operator, then formulate the sub-questions (unless the operator is \"Scan\", in which case no sub-questions are needed and this step must be skipped), and finally justify the decomposition.\n"
            + "Provide your reasoning enclosed in <think> and </think> tags, and afterwards provide the final answer in the following format: the first line of the final answer should be the toplevel operator, the following lines should be the predicted sub-questions."
        )

    def to_chat_template(self, example) -> ChatML:
        db_id = example['db_id']

        user = self.user_prompt(db_id, example['question'])

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

            return ChatML(
                messages=[
                    Message(role="system", content=self.system_prompt),
                    Message(role="user", content=user),
                    Message(role="assistant", content=response),
                ]
            )
        else:
            # user += "\n- **The decomposition must adhere to the schema** - for example, if the relevant information to answer the question is spread across diffrent tables, the \"Scan\" operator is not a good fit for this question."
            # user += "\n- **DO NOT omit columns in the sub-questions** - combining the sub questions (if any) with the operator must result in all the columns requested by the question. While formulating the sub-questions, you must ensure that the exact columns requested by the question are present in the final result."
            # user += "\n- **Carefully evaluate your decomposition** - if the sub-questions require adjustments to fully align with the schema, the original question, and complement each other, reflect on your decomposition and present the revised sub-questions to be in your final answer."
            return ChatML(
                messages=[
                    Message(role="system", content=self.system_prompt),
                    Message(role="user", content=user),
                ]
            )
