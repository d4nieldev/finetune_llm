import json
import asyncio

from src.utils.qpl.tree import Operator
from src.utils.qpl.schema import DBSchema, SchemaRepresentation
from src.utils.qpl.tree import QPLQDTree

import dspy

class Decompose(dspy.Signature):
    """
    Given a database schema and a question in natural language, you must predict the toplevel QPL operator and if needed, decompose the input question into one or two simpler sub-questions which describe the arguments of the toplevel operator.

The toplevel QPL operators are:
- **Scan** - Scan all rows in a table with optional filtering predicate (no decomposition needed - the question is atomic)
- **Aggregate** - Aggregate a stream of tuples, optionally using a grouping criterion into a stream of groups (1 sub-question)
- **Filter** - Remove tuples from a stream that do not match a predicate (1 sub-question)
- **Sort** - Sort a stream according to a sorting expression (1 sub-question)
- **TopSort** - Select the top-K tuples from a stream according to a sorting expression (1 sub-question)
- **Join** - Perform a logical join operation between two streams based on a join condition (2 sub-questions)
- **Except** - Compute the set difference between two streams of tuples (2 sub-questions)
- **Intersect** - Compute the set intersection between two streams of tuples (2 sub-questions)
- **Union** - Compute the set union between two streams of tuples (2 sub-questions)
    """

    db_schema: str = dspy.InputField(desc="The schema of the database the question is being asked about")
    question: str = dspy.InputField(desc="The question to be decomposed")

    operator: Operator = dspy.OutputField(desc="The top-level operator of the question")
    sub_questions: list[str] = dspy.OutputField(desc="The sub-questions that, when combined with the operator, form the given question")


class Complete(dspy.Signature):
    """
    Given a database schema, a QPL query prefix, and a natural language question, complete the final line of the query so it completes the user request.

QPL is a formalism used to describe data retrieval operations over an SQL schema in a modular manner.
A QPL plan is a sequence of instructions for querying tabular data to answer a natural language question.

Below is the formal specification for each operation in valid QPL:
<qpl> ::= <line>+
<line> ::= #<integer> = <qpl-operator>
<qpl-operator> ::= <scan> | <aggregate> | <filter> | <sort> | <topsort> | <join> | <except> | <intersect> | <union>

-- Leaf operator
<scan> ::= Scan Table [ <table-name> ] <pred-non-qualif>? <distinct>? <output-non-qualif>

-- Unary operators
<aggregate> ::= Aggregate [ <input> ] <group-by>? <output-agg>
<filter> ::= Filter [ <input> ] <pred-non-qualif> <distinct>? <output-non-qualif>
<sort> ::= Sort [ <input> ] <order-by> <output-non-qualif>
<topsort> ::= TopSort [ <input> ] Rows [ <number> ] <order-by> <withTies>? <output-non-qualif>

-- Binary operators
<join> ::= Join [ <input> , <input> ] <pred-qualif>? <distinct>? <output-qualif>
<except> ::= Except [ <input> , <input> ] <pred-qualif> <output-qualif>
<intersect> ::= Intersect [ <input> , <input> ] <pred-qualif> <output-qualif>
<union> ::= Union [ <input> , <input> ] <output-qualif>

-- General
<group-by> ::= GroupBy [ <column-name> (, <column-name>)* ]
<order-by> ::= OrderBy [ <column-name> <direction> (, <column-name> <direction>)* ]
<withTies> ::= WithTies [ true | false ]
<direction> ::= ASC | DESC
<pred-non-qualif> ::= Predicate [ <comparison> (AND | OR <comparison>)* ]
<comparison-non-qualif> ::= <column-name> <operator> <value>
<pred-qualif> ::= Predicate [ <comparison-qualif> (AND | OR <comparison-qualif>)* ]
<comparison-qualif> ::= <qualif-column-name> <operator> (<value> | <qualif-column-name>)
<distinct> ::= Distinct [ true | false ]
<output-non-qualif> ::= Output [ <column-name> (, <column-name>)* ]
<output-agg> ::= Output [ <agg-column-name> (, <agg-column-name>)* ]
<output-qualif> ::= Output [ <qualif-column-name> (, <qualif-column-name>)* ]
<agg-column-name> ::= countstar | <agg-func>(<column-name>) | <agg-func>(DISTINCT <column-name>)
<agg-func> ::= COUNT | SUM | AVG | MIN | MAX
<qualif-column-name> ::= #<number>.<column-name>

Using valid QPL, complete the last line in order to answer the question for the database schema provided below.
    """

    db_schema: str = dspy.InputField(desc="The schema of the database the question is being asked about")
    question: str = dspy.InputField(desc="The question to be answered by a QPL query")
    prefix_qpl: str = dspy.InputField(desc="The prefix of the QPL query that needs to be completed")

    last_line: str = dspy.OutputField(desc="The completed last line of the QPL query")


def post_order_index_tree(tree: QPLQDTree, counter: int = 1) -> int:
    for child in tree.children:
        counter = post_order_index_tree(child, counter)
    tree.line_num = counter
    return counter + 1



class RecursiveDecomposer(dspy.Module):
    def __init__(self, cot: bool = True, schema_repr: SchemaRepresentation = SchemaRepresentation.M_SCHEMA):
        self.model = dspy.ChainOfThought(Decompose) if cot else dspy.Predict(Decompose)
        self.cot = cot
        self.schema_repr = schema_repr

    async def aforward(self, db_schema: DBSchema, question: str) -> QPLQDTree:
        async def rec(q: str, _parent: QPLQDTree | None = None) -> QPLQDTree:
            # 1) predict operator + sub_questions for this node
            output = await self.model.acall(
                db_schema=db_schema.ddl() if self.schema_repr == SchemaRepresentation.DDL else db_schema.m_schema(),
                question=q
            )

            # 2) create node
            tree = QPLQDTree(
                question=q,
                db_id=db_schema.db_id,
                op=output.operator,
                decomposition_cot=getattr(output, "reasoning", None) if self.cot else None,
                parent=_parent
            )

            # 3) kick off child decompositions concurrently
            tasks = [rec(sq, tree) for sq in output.sub_questions]
            children = await asyncio.gather(*tasks) if tasks else []
            tree.children = tuple(children)
            return tree

        root = await rec(question)
        post_order_index_tree(root)
        return root

    def forward(self, db_schema: DBSchema, question: str) -> QPLQDTree:
        return asyncio.run(self.aforward(db_schema=db_schema, question=question))


class RecursiveCompleter(dspy.Module):
    def __init__(self, cot: bool = True, schema_repr: SchemaRepresentation = SchemaRepresentation.M_SCHEMA):
        self.model = dspy.ChainOfThought(Complete) if cot else dspy.Predict(Complete)
        self.schema_repr = schema_repr

    async def aforward(self, tree: QPLQDTree, db_schema: DBSchema) -> None:
        async def rec(node: QPLQDTree) -> None:
            # 1) complete all children concurrently
            if node.children:
                await asyncio.gather(*(rec(c) for c in node.children))

            # 2) build prefix from (now completed) children
            prefix_qpl = "\n".join(child.qpl for child in node.children)

            # 3) add the partial line to complete
            if node.op == Operator.SCAN:
                children_str = "Table"
            else:
                children_str = f"[ {', '.join(f'#{c.line_num}' for c in node.children)} ]"
            prefix_qpl += f"#{node.line_num} = {node.op} {children_str} ..."

            # 4) complete current node’s line
            output = await self.model.acall(
                db_schema=db_schema.ddl() if self.schema_repr == SchemaRepresentation.DDL else db_schema.m_schema(),
                question=node.question,
                prefix_qpl=prefix_qpl
            )
            node.qpl_line = output.last_line

        await rec(tree)

    def forward(self, tree: QPLQDTree, db_schema: DBSchema) -> None:
        return asyncio.run(self.aforward(tree=tree, db_schema=db_schema))


class TextToQPL(dspy.Module):
    def __init__(self, cot: bool = True, schema_repr: SchemaRepresentation = SchemaRepresentation.M_SCHEMA):
        self.decomposer = RecursiveDecomposer(cot=cot, schema_repr=schema_repr)
        self.completer = RecursiveCompleter(cot=cot, schema_repr=schema_repr)
        self.db_schemas = DBSchema.from_db_schemas_file()

    async def aforward(self, db_id: str, question: str) -> QPLQDTree:
        db_schema = self.db_schemas[db_id]
        tree = await self.decomposer.acall(db_schema=db_schema, question=question)
        await self.completer.acall(tree=tree, db_schema=db_schema)
        return tree

    # keep a sync wrapper if you like
    def forward(self, db_id: str, question: str) -> QPLQDTree:
        return asyncio.run(self.aforward(db_id=db_id, question=question))


if __name__ == "__main__":
    dspy.settings.configure(lm=dspy.LM("ollama_chat/gpt-oss"))
    text_to_qpl = TextToQPL()

    async def main():
        tree = await text_to_qpl.acall(
            db_id="car_1",
            question="Which models are lighter than 3500 but not built by the 'Ford Motor Company'?"
        )
        with open('out.json', 'w') as f:
            json.dump(tree.to_dict(), f, indent=2)

    asyncio.run(main())