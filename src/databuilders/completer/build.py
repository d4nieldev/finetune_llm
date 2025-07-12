from typing import Tuple, List, Optional, Dict
import logging as log
from collections import defaultdict
from copy import deepcopy
from itertools import product

from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

from src.utils.qpl.tree import QPLTree, PartialQDTree, Operator


class CompleterDatasetError(Exception):
    pass

class OperatorMismatchError(CompleterDatasetError):
    def __init__(self, decomposer_op: str, qpl_op: str, sub_question: str):
        super().__init__(f"Operator mismatch: Operator in decomposer dataset ({decomposer_op}) does not match operator in nl2sql dataset ({qpl_op}) for sub-question {sub_question!r}. ")

class ChildrenMismatchError(CompleterDatasetError):
    def __init__(self, qd_children: int, qpl_children: int):
        super().__init__(f"Children mismatch: Number of children in decomposer dataset ({qd_children}) should be less then or equal to number of children in the nl2sql dataset ({qpl_children})")



def get_decomposer_roots(decomposer_data: Dataset, root_questions: set[str]) -> List[PartialQDTree]:
    # A question might have multiple decompositions, so we need to group them by question and db_id
    q_to_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in decomposer_data:
        question = row["question"]
        db_id = row["db_id"]
        q_to_rows[(question, db_id)].append(row)

    def construct_trees(row) -> list[PartialQDTree]:
        question = row["question"]
        db_id = row["db_id"]
        op = Operator(row['op'])
        
        # The question and its children might have multiple decompositions, so we need to construct combinations of children
        child_i_options = [[], []]
        for i, sq_key in enumerate(['sub_question_1', 'sub_question_2']):
            if (sq := row[sq_key]):
                if (sq, db_id) not in q_to_rows:
                    raise ValueError(
                        f"Sub-question {sq!r} not found in decomposer data for question {question!r} with db_id {db_id!r}"
                    )
                for sq_row in q_to_rows[(sq, db_id)]:
                    sq_child_options = construct_trees(sq_row)
                    child_i_options[i].extend(sq_child_options)
        
        # For each combination of children, create a diffrent tree
        child_i_options = [options if options else [None] for options in child_i_options]
        children_combinations = list(product(*child_i_options))
        trees = []
        for children in children_combinations:
            real_children = tuple([child for child in children if child is not None])
            tree = PartialQDTree(
                question=question,
                db_id=db_id,
                op=op,
                children=real_children,
            )
            for child in real_children:
                child.parent = tree
            trees.append(tree)

        return trees

    all_trees = [
        tree 
        for row in tqdm(decomposer_data, desc="Constructing QD trees") 
        for tree in construct_trees(row)
        if row["question"] in root_questions  # only take root questions
    ]

    return [tree for tree in all_trees if tree.parent is None]


def get_qpl_trees(nl2qpl_data: Dataset) -> Dict[Tuple[str, Operator, str], QPLTree]:
    q_to_qpl_tree: Dict[Tuple[str, Operator, str], QPLTree] = {}

    # create QPL trees
    for row in tqdm(nl2qpl_data, desc="Creating QPL trees from NL2QPL data"):
        question = row["question"]
        qpl = row["qpl"]
        db_id, qpl_code = [item.strip() for item in qpl.split("|")]
        qpl_rows = [qpl_row.strip() for qpl_row in qpl_code.split(";")]
        op = qpl_rows[-1].split(' = ')[1].split(' ')[0].strip()
        
        if db_id == 'car_1':
            db_id = 'car_11'
        q_to_qpl_tree[(question, Operator(op), db_id)] = QPLTree.from_qpl_lines(qpl_rows)

    return q_to_qpl_tree


def complete_trees_qpl(
    decomposer_roots: List[PartialQDTree],
    q_to_qpl_tree: Dict[Tuple[str, Operator, str], QPLTree]
) -> None:

    def complete_tree(root: PartialQDTree, qpl_tree: QPLTree) -> None:
        if root.op != qpl_tree.op:
            raise OperatorMismatchError(root.op.value, qpl_tree.op.value, root.question)
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
            qpl_tree = q_to_qpl_tree[(root.question, root.op, root.db_id)]
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
    decomposer_data = [row for row in decomposer_data if row['question'] not in [row['sub_question_1'], row['sub_question_2']]]
    root_questions = set(row['question'] for row in nl2qpl_data)
    root_qd_trees = get_decomposer_roots(decomposer_data, root_questions)
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