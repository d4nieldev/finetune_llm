import re
from dataclasses import dataclass
from typing import List, Optional, Union

from data.create_composer_data import Operator, Tuple


@dataclass
class QPLTree:
    qpl_row: str = None  # type: ignore
    children: Optional[Union[Tuple["QPLTree"], Tuple["QPLTree", "QPLTree"]]] = None

    @property
    def op(self) -> Operator:
        return Operator(self.qpl_row.split("=")[1].strip().split(" ")[0].strip())

    @property
    def children_qpl(self) -> str:
        if self.children is None:
            return ""
        return "\n".join([(child.children_qpl + "\n" + child.qpl_row).strip() for child in self.children]).replace("\n", " ; ")


def get_qpl_tree(qpl_lines: List[str]) -> QPLTree:
    row_nodes = [QPLTree() for _ in qpl_lines]
    for qpl_row in qpl_lines:
        line_numbers = [int(match) for match in re.findall(r"#(\d+)", qpl_row)]
        row_id = line_numbers[0] - 1
        children = [row_nodes[line_num - 1] for line_num in set(line_numbers[1:])]
        row_nodes[row_id].qpl_row = qpl_row
        row_nodes[row_id].children = tuple(children) if children else None  # type: ignore
    return row_nodes[-1]
