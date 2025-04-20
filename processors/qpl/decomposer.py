import json
from .nl2qpl import NL2QPLProcessor


class QPLDecomposerProcessor(NL2QPLProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        q_to_id = {}
        for id, content in self._db_content.items():
            q_to_id[content['question']] = id
        self._q_to_id = q_to_id
    
    def process_row(self, row):
        db_id = self._db_content[row['id']]["db_id"]

        # get id
        id = self._q_to_id.get(row['question'])
        if id is None:
            # go to parent
            for dataset in self._dataset.values():
                for r in dataset:
                    if row['question'] == r['sub_question_1'] or row['question'] == r['sub_question_2']:
                        id = self._q_to_id[row['question']]
                        break

        prompt = (
            "Given a database schema and a question in natural language, "
            + "you must predict the toplevel operator and if needed, decompose the input question into one or two "
            + "simpler sub-questions which describe the arguments of the toplevel operator.\n\n"

            + "The toplevel operators are:\n"
            + "**Scan** - Scan all rows in a table with optional filtering predicate (no decomposition needed - the question is atomic)\n"
            + "**Aggregate** - Aggregate a stream of tuples using a grouping criterion into a stream of groups (1 subquestion)\n"
            + "**Filter** - Remove tuples from a stream that do not match a predicate (1 subquestion)\n"
            + "**Sort** - Sort a stream according to a sorting expression (1 subquestion)\n"
            + "**TopSort** - Select the top-K tuples from a stream according to a sorting expression (1 subquestion)\n"
            + "**Join** - Perform a logical join operation between two streams based on a join condition (2 subquestions)\n"
            + "**Except** - Compute the set difference between two streams of tuples (2 subquestions)\n"
            + "**Intersect** - Compute the set intersection between two streams of tuples (2 subquestions)\n"
            + "**Union** - Compute the set union between two streams of tuples (2 subquestions)\n\n"

            + f"Database Name: {db_id}\n\n"

            + "Database Schema:\n"
            + f"```DDL\n{self._create_table_prompt(id)}```\n\n"

            + f"""Question: {row["question"].strip()}\n\n"""

            + "Your output should be a JSON object with the following keys:\n"
            + "- \"toplevel_operator\" (string) - the toplevel operator (one of the above)\n"
            + "- \"subquestions\" (list of 0, 1, or 2 strings) - the list of subquestions. The list size depends on the selected toplevel operator\n"
        )

        subquestions_lst = []
        if row['sub_question_1']:
            subquestions_lst.append(row['sub_question_1'])
        if row['sub_question_2']:
            subquestions_lst.append(row['sub_question_2'])

        indent = ' ' * 4
        response = '{\n%s"toplevel_operator": "%s",\n%s"subquestions": %s\n}' % (
            indent,
            row["toplevel_operator"],
            indent,
            f'\n{indent}'.join(json.dumps(subquestions_lst, indent=len(indent)).splitlines()),
        )

        return {"prompt": prompt, "response": response}
