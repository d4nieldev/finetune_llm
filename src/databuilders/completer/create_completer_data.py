from typing import Tuple, List, Optional, Dict
import logging as log

from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

from src.utils.qpl.tree import QPLTree, PartialQDTree, Operator


class CompleterDatasetError(Exception):
    pass

class OperatorMismatchError(CompleterDatasetError):
    def __init__(self, decomposer_op: str, qpl_op: str):
        super().__init__(f"Operator mismatch: Operator in decomposer dataset ({decomposer_op}) does not match operator in nl2sql dataset ({qpl_op})")

class ChildrenMismatchError(CompleterDatasetError):
    def __init__(self, qd_children: int, qpl_children: int):
        super().__init__(f"Children mismatch: Number of children in decomposer dataset ({qd_children}) should be less then or equal to number of children in the nl2sql dataset ({qpl_children})")



def get_decomposer_roots(decomposer_data: Dataset) -> List[PartialQDTree]:
    q_to_tree: dict[Tuple[str, str], PartialQDTree] = {}

    def get_q_tree(question: Optional[str], db_id: str) -> Optional[PartialQDTree]:
        """Returns the question tree for the given question, creating a leaf node if it doesn't exist."""
        if question is None:
            return None
        if db_id == 'car_11':
            db_id = 'car_1'
        if (question, db_id) not in q_to_tree:
            q_to_tree[(question, db_id)] = PartialQDTree(question=question, db_id=db_id)
        return q_to_tree[(question, db_id)]

    def construct_tree(row) -> PartialQDTree:
        question = row["question"]
        db_id = row["db_id"]
        tree = get_q_tree(question, db_id)

        if tree is None:
            raise ValueError(f"Tree for question {question} not found in q_to_tree")

        tree.op = Operator(row['op'])

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
        question = row["question"]
        qpl = row["qpl"]
        db_id, qpl_code = [item.strip() for item in qpl.split("|")]
        qpl_rows = [qpl_row.strip() for qpl_row in qpl_code.split(";")]

        q_to_qpl_tree[(question, db_id)] = QPLTree.from_qpl_lines(qpl_rows)

    return q_to_qpl_tree


def complete_trees_qpl(
    decomposer_roots: List[PartialQDTree],
    q_to_qpl_tree: Dict[Tuple[str, str], QPLTree]
) -> None:

    def complete_tree(root: PartialQDTree, qpl_tree: QPLTree) -> None:
        if root.op != qpl_tree.op:
            raise OperatorMismatchError(root.op.value, qpl_tree.op.value)
        if len(root.children) > len(qpl_tree.children):
            raise ChildrenMismatchError(len(root.children), len(qpl_tree.children))

        root.qpl_line = qpl_tree.qpl_line
        root.prefix_qpl = qpl_tree.prefix_qpl

        # change order when necessary
        if len(root.children) == 2 and root.children[0].op != qpl_tree.children[0].op:
            qpl_tree.children = qpl_tree.children[1], qpl_tree.children[0]

        for root_child, qpl_child in zip(root.children, qpl_tree.children):
            complete_tree(root_child, qpl_child)

    for root in tqdm(decomposer_roots, desc="Completing trees with QPL"):
        try:
            qpl_tree = q_to_qpl_tree[(root.question, root.db_id)]
            complete_tree(root, qpl_tree)
        except (KeyError, CompleterDatasetError) as e:
            msg = f"Error for question {root.question!r} with db_id {root.db_id!r}:"
            if isinstance(e, KeyError):
                msg += f"\n\tQuestion could not be found in NL2QPL data."
            else:
                msg += f"\n\t{e}"
            log.warning(msg)


def create_completer_dataset(decomposer_roots: List[PartialQDTree]) -> List[Dict]:
    def get_tree_rows(root: PartialQDTree) -> List[Dict]:
        rows_to_return = [child_row for child in root.children for child_row in get_tree_rows(child)]

        if root.prefix_qpl is not None and root.qpl_line is not None:
            if root.parent is not None and root.parent.prefix_qpl is not None:
                # enrich parent prefix QPL with child question
                root.parent.prefix_qpl = root.parent.prefix_qpl.replace(
                    root.qpl_line + " ;",
                    f"{root.qpl_line} ; -- {root.question}"
                )
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


def load_qd_trees(
        nl2qpl_dataset_id: str = "d4nieldev/nl2qpl-ds",
        decomposer_dataset_id: str = "bgunlp/question_decomposer_ds",
        split: str = "train"
    ) -> List[PartialQDTree]:
    # Load the NL2QPL data and create QPL tree for each question
    nl2qpl_data = load_dataset(nl2qpl_dataset_id, split=split)
    qpl_trees = get_qpl_trees(nl2qpl_data)
    log.info(f"Number of QPL trees in {split}: {len(qpl_trees)}")

    # Load the decomposer data and create partial QD trees
    decomposer_data = load_dataset(decomposer_dataset_id, split=split)
    root_qd_trees = get_decomposer_roots(decomposer_data)
    log.info(f"Number of partial QD trees in {split}: {len(root_qd_trees)}")

    # Complete the QD trees with QPL data
    complete_trees_qpl(root_qd_trees, qpl_trees)

    return root_qd_trees


if __name__ == "__main__":
    nl2qpl_dataset_id = "d4nieldev/nl2qpl-ds"
    decomposer_dataset_id = "bgunlp/question_decomposer_ds"

    log.basicConfig(
        level=log.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    dataset = {}

    for split in ["train", "validation"]:
        # merge datasets to create combined QD trees
        root_qd_trees = load_qd_trees(
            nl2qpl_dataset_id=nl2qpl_dataset_id,
            decomposer_dataset_id=decomposer_dataset_id,
            split=split
        )

        # Create dataset rows
        split_data = create_completer_dataset(root_qd_trees)

        dataset[split] = split_data

    # Upload the dataset to Hugging Face
    ds = DatasetDict({
        split: Dataset.from_list(data)
        for split, data in dataset.items()
    })
    ds.push_to_hub("d4nieldev/qpl-completer-ds")