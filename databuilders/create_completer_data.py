from typing import Tuple, List, Optional, Dict

from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

from utils.qpl.tree import QPLTree, QPLQDTree, Operator


def get_decomposer_roots(decomposer_data: Dataset) -> List[QPLQDTree]:
    q_to_tree: dict[Tuple[str, str], QPLQDTree] = {}

    def get_q_tree(question: Optional[str], db_id: str) -> Optional[QPLQDTree]:
        """Returns the question tree for the given question, creating a leaf node if it doesn't exist."""
        if question is None:
            return None
        if (question, db_id) not in q_to_tree:
            q_to_tree[(question, db_id)] = QPLQDTree(question=question, db_id=db_id)
        return q_to_tree[(question, db_id)]

    def construct_tree(row) -> QPLQDTree:
        question = row["question"]
        db_id = row["db_id"]
        tree = get_q_tree(question, db_id)

        if tree is None:
            raise ValueError(f"Tree for question {question} not found in q_to_tree")

        if row['op'] == 'Top':
            row['op'] = 'TopSort'
        tree.op = Operator(row["op"])

        child1 = get_q_tree(row['sub_question_1'], db_id)
        child2 = get_q_tree(row['sub_question_2'], db_id)
        if child1 is not None:
            child1.parent = tree
        if child2 is not None:
            child2.parent = tree
        tree.children = tuple([child for child in [child1, child2] if child is not None])

        return tree

    all_trees = [construct_tree(row) for row in tqdm(decomposer_data, desc="Constructing QD trees")]

    return [tree for tree in all_trees if tree.parent is None]


def get_qpl_trees(nl2qpl_data: Dataset) -> Dict[Tuple[str, str], QPLTree]:
    q_to_qpl_tree: Dict[Tuple[str, str], QPLTree] = {}

    # create QPL trees
    for row in tqdm(nl2qpl_data, desc="Creating QPL trees from NL2QPL data"):
        question = row["question"]  # type: ignore
        qpl = row["qpl"]  # type: ignore
        db_id, qpl_code = [item.strip() for item in qpl.split("|")]
        qpl_rows = [qpl_row.strip() for qpl_row in qpl_code.split(";")]

        q_to_qpl_tree[(question, db_id)] = QPLTree.from_qpl_lines(qpl_rows)

    return q_to_qpl_tree


def complete_trees_qpl(
    decomposer_roots: List[QPLQDTree],
    q_to_qpl_tree: Dict[Tuple[str, str], QPLTree]
) -> None:

    def complete_tree(qpl_qd_tree: QPLQDTree, qpl_tree: QPLTree) -> None:
        assert len(qpl_qd_tree.children) <= len(qpl_tree.children), f"Number of children in QD tree ({len(qpl_qd_tree.children)}) should be less then or equal to number of children in QPL tree ({len(qpl_tree.children)})"
        assert qpl_qd_tree.op == qpl_tree.op, f"Operator in QD tree ({qpl_qd_tree.op.value}) does not match QPL tree ({qpl_tree.op.value})"

        qpl_qd_tree.qpl_line = qpl_tree.qpl_row

        # change order if necessary
        if len(qpl_qd_tree.children) == 2 and qpl_qd_tree.children[0].op != qpl_tree.children[0].op:
            qpl_tree.children = qpl_tree.children[1], qpl_tree.children[0]

        for root_child, qpl_child in zip(qpl_qd_tree.children, qpl_tree.children):
            complete_tree(root_child, qpl_child)

    for root in tqdm(decomposer_roots, desc="Completing trees with QPL"):
        try:
            qpl_tree = q_to_qpl_tree[(root.question, root.db_id)]
        except KeyError:
            print(f"Decomposer question {(root.question, root.db_id)} could not be found in NL2QPL data.")
            continue

        try:
            complete_tree(root, qpl_tree)
        except Exception as e:
            print(f"Exception occured for decomposer question \"{root.question}\"")
            print(f"\t{e}")
            continue


def create_completer_dataset(decomposer_roots: List[QPLQDTree]) -> List[Dict]:
    def get_tree_rows(root: QPLQDTree) -> List[Dict]:
        rows_to_return = []
        if root.children:
            rows_to_return += [child_row for child in root.children for child_row in get_tree_rows(child)]

        if root.prefix_qpl is not None and root.qpl_line is not None:
            rows_to_return.append(
                {
                    "db_id": root.db_id,
                    "parent_question": root.parent.question if root.parent else None,
                    "question": root.question,
                    "prefix_qpl": root.prefix_qpl,
                    "op": root.op.value,
                    "qpl_line": root.qpl_line
                }
            )

        return rows_to_return

    all_rows = [
        root_row
        for root in tqdm(decomposer_roots, desc="Creating dataset rows")
        for root_row in get_tree_rows(root)
    ]

    return all_rows


if __name__ == "__main__":
    nl2qpl_dataset_id = "d4nieldev/nl2qpl-ds"
    decomposer_dataset_id = "bgunlp/question_decomposer_ds"

    dataset = {}

    for split in ["train", "validation"]:
        # Load the NL2QPL data and create QPL tree for each question
        nl2qpl_data = load_dataset(nl2qpl_dataset_id, split=split)
        qpl_trees = get_qpl_trees(nl2qpl_data)  # type: ignore
        print(f"Number of QPL trees: {len(qpl_trees)}")

        # Load the decomposer data and create partial QD trees
        decomposer_data = load_dataset(decomposer_dataset_id, split=split)
        root_qd_trees = get_decomposer_roots(decomposer_data)  # type: ignore
        print(f"Number of partial QD trees: {len(root_qd_trees)}")

        # Complete the QD trees with QPL data
        complete_trees_qpl(root_qd_trees, qpl_trees)

        # Create dataset rows
        split_data = create_completer_dataset(root_qd_trees)

        dataset[split] = split_data

    # Upload the dataset to Hugging Face
    ds = DatasetDict({
        split: Dataset.from_list(data)
        for split, data in dataset.items()
    })
    ds.push_to_hub("d4nieldev/qpl-completer-ds")
