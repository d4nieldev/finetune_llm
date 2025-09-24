import re
import logging

from datasets import Dataset, DatasetDict

from src.utils.tree import PartialQDTree, Operator
from src.databuilders.completer.build import load_qd_trees


def merge_filter_scan(filter_line: str, scan_line: str) -> str:
    scan_pat = r"#(?P<idx>\d+) = Scan Table \[ (?P<table>\w+) \]( Predicate \[ (?P<pred>[^\]]+) \])?( Distinct \[ (?P<distinct>true) \])? Output \[ (?P<out>[^\]]+) \]"
    filter_pat = r"#(?P<idx>\d+) = Filter \[ (?P<child>#\d+) \] Predicate \[ (?P<pred>[^\]]+) \]( Distinct \[ (?P<distinct>true) \])? Output \[ (?P<out>[^\]]+) \]"

    if not (scan_m := re.match(scan_pat, scan_line)):
        raise ValueError(f"Scan line does not match expected pattern: {scan_line}")
    if not (filter_m := re.match(filter_pat, filter_line)):
        raise ValueError(f"Filter line does not match expected pattern: {filter_line}")

    idx = filter_m.group("idx")
    table = scan_m.group("table")
    scan_pred = scan_m.group("pred")
    filter_pred = filter_m.group("pred")
    distinct = scan_m.group("distinct") or filter_m.group("distinct")
    out = filter_m.group("out")

    if scan_pred:
        composite_scan_pred = set(scan_pred.lower().split(' ')).intersection({'and', 'or'})
        composite_filter_pred = set(filter_pred.lower().split(' ')).intersection({'and', 'or'})
        filter_pred = '( ' + filter_pred + ' )' if composite_filter_pred else filter_pred
        scan_pred = '( ' + scan_pred + ' )' if composite_scan_pred else scan_pred
        pred = f"{scan_pred} AND {filter_pred}"
    else:
        pred = filter_pred

    merged_line = f"#{idx} = Scan Table [ {table} ] Predicate [ {pred} ]"
    if distinct:
        merged_line += " Distinct [ true ]"
    merged_line += f" Output [ {out} ]"

    return merged_line


def fix_tree(qd_tree: PartialQDTree) -> None:
    if not qd_tree.is_complete():
        # already logged in `load_qd_trees`
        # logging.warning(f"QDTree is not complete: {qd_tree.question}")
        return
    
    def merge(qd_tree: PartialQDTree) -> None:
        for child in qd_tree.children:
            merge(child)
        if qd_tree.op == Operator.FILTER and qd_tree.children[0].op == Operator.SCAN:
            qd_tree.op = Operator.SCAN
            qd_tree.prefix_qpl = ''
            if not qd_tree.qpl_line:
                raise ValueError(f"QPL line is not set for node with question: {qd_tree.question}")
            if not qd_tree.children[0].qpl_line:
                raise ValueError(f"QPL line is not set for child node with question: {qd_tree.children[0].question}")
            scan_line = qd_tree.children[0].qpl_line
            filter_line = qd_tree.qpl_line
            qd_tree.qpl_line = merge_filter_scan(filter_line=filter_line, scan_line=scan_line)
            qd_tree.children = ()

            parent = qd_tree.parent
            while parent:
                if not parent.prefix_qpl:
                    raise ValueError(f"Prefix QPL is not set for parent node with question: {parent.question}")
                scan_filter_pat = scan_line+" ;\n"+filter_line+" ;"
                new_prefix = re.sub(re.escape(scan_filter_pat), qd_tree.qpl_line+" ;",  parent.prefix_qpl)
                if new_prefix == parent.prefix_qpl:
                    raise ValueError(f"Prefix QPL did not change for parent node with question: {parent.question}")
                parent.prefix_qpl = new_prefix
                parent = parent.parent
    
    def find_idx_successor(qd_tree: PartialQDTree, idx: int) -> int | None:
        "Returns the index of the node with the smallest index greater or equal to `idx`. None if no such node exists."
        if qd_tree.idx == idx:
            return qd_tree.idx
        if qd_tree.idx > idx:
            for child in qd_tree.children:
                if (successor_idx := find_idx_successor(child, idx)) is not None:
                    return successor_idx
            return qd_tree.idx
        return None

    def rename_idx(qd_tree: PartialQDTree, old_idx: int, new_idx: int) -> None:
        pattern = re.compile(r'(?<!\d)#' + str(old_idx) + r'(?!\d)')
        def rec(qd_tree: PartialQDTree) -> None:
            if not qd_tree.qpl_line:
                raise ValueError(f"QPL line is not set for node with question: {qd_tree.question}")
            qd_tree.qpl_line = pattern.sub(f"#{new_idx}", qd_tree.qpl_line)
            if qd_tree.prefix_qpl:
                qd_tree.prefix_qpl = pattern.sub(f"#{new_idx}", qd_tree.prefix_qpl)
            for child in qd_tree.children:
                rec(child)
        rec(qd_tree)

    def renumber(qd_tree: PartialQDTree) -> None:
        idx = 1
        while (successor_idx := find_idx_successor(qd_tree, idx)) is not None:
            if successor_idx != idx:
                rename_idx(qd_tree, old_idx=successor_idx, new_idx=idx)
            idx += 1

    merge(qd_tree)
    renumber(qd_tree)


def get_examples(split: str = "validation") -> list[dict]:
    qd_trees = load_qd_trees(split=split)
    for qd_tree in qd_trees: fix_tree(qd_tree)

    def tree_rows(qd_tree: PartialQDTree) -> list[dict]:
        rows = [child_row for child in qd_tree.children for child_row in tree_rows(child)]

        if qd_tree.prefix_qpl is not None and qd_tree.qpl_line is not None and qd_tree.op is not None:
            parent = qd_tree.parent
            while parent and parent.prefix_qpl:
                # enrich parent prefix QPL with child question
                parent.prefix_qpl = parent.prefix_qpl.replace(
                    qd_tree.qpl_line + " ;",
                    f"{qd_tree.qpl_line} ; -- {qd_tree.question}"
                )
                parent = parent.parent
            rows.append(
                {
                    "db_id": qd_tree.db_id,
                    "parent_question": qd_tree.parent.question if qd_tree.parent else None,
                    "question": qd_tree.question,
                    "op": qd_tree.op.value,
                    "sub_question_1": qd_tree.children[0].question if len(qd_tree.children) > 0 else None,
                    "sub_question_2": qd_tree.children[1].question if len(qd_tree.children) > 1 else None,
                    "prefix_qpl": qd_tree.prefix_qpl,
                    "qpl_line": qd_tree.qpl_line
                }
            )

        return rows
    
    return [
        row
        for qd_tree in qd_trees
        for row in tree_rows(qd_tree)
    ]


def main():
    dataset = {}
    for split in ["train", "validation"]:
        dataset[split] = get_examples(split=split)
    
    ds = DatasetDict({
        split: Dataset.from_list(data)
        for split, data in dataset.items()
    })
    ds.push_to_hub("d4nieldev/qpl-decomposer-completer-ds")


if __name__ == "__main__":
    main()
