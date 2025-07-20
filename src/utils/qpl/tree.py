import re
from enum import StrEnum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


class Operator(StrEnum):
    SCAN = "Scan"
    AGGREGATE = "Aggregate"
    FILTER = "Filter"
    SORT = "Sort"
    TOP = "Top"
    TOPSORT = "TopSort"
    JOIN = "Join"
    EXCEPT = "Except"
    INTERSECT = "Intersect"
    UNION = "Union"


@dataclass
class QPLTree:
    """A tree structure representing a QPL query."""
    qpl_line: str = None  # type: ignore
    children: Tuple["QPLTree", ...] = ()

    @property
    def op(self) -> Operator:
        return Operator(self.qpl_line.split("=")[1].strip().split(" ")[0].strip())

    @property
    def prefix_qpl(self) -> str:
        return "\n".join([child.qpl for child in sorted(self.children, key=lambda x: x.line_num)])
    
    @property
    def line_num(self) -> int:
        """Extracts the line number from the QPL row."""
        match = re.search(r"#(\d+)", self.qpl_line)
        if match:
            return int(match.group(1))
        raise ValueError(f"Line number not found in QPL row: {self.qpl_line}")
    
    @property
    def qpl(self) -> str:
        return (self.prefix_qpl + "\n" + self.qpl_line + " ;").strip()
    
    @staticmethod
    def from_qpl_lines(qpl_lines: List[str]) -> "QPLTree":
        """Construct a QPLTree from a list of QPL lines."""
        row_nodes = defaultdict(lambda: QPLTree())
        for qpl_row in qpl_lines:
            line_numbers = [int(match) for match in re.findall(r"#(\d+)", qpl_row)]
            row_id = line_numbers[0] - 1
            children = [row_nodes[line_num - 1] for line_num in list(dict.fromkeys(line_numbers[1:]))]  # preserve children order!
            row_nodes[row_id].qpl_line = qpl_row
            row_nodes[row_id].children = tuple(children)
        root_row_id = max(row_nodes.keys())
        return row_nodes[root_row_id]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_num": self.line_num,
            "op": self.op,
            "qpl_line": self.qpl_line,
            "qpl": self.qpl,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class QPLQDTree:
    """A tree structure representing a QPL query with full decomposition - used in inference."""
    question: str
    db_id: str
    op: Operator = None   # type: ignore
    line_num: int = None  # type: ignore
    qpl_line: str = None  # type: ignore
    parent: Optional["QPLQDTree"] = None
    children: Tuple["QPLQDTree", ...] = ()
    decomposition_cot: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        output = {
            "db_id": self.db_id,
            "question": self.question,
            "is_valid": self.is_valid,
            "line_num": self.line_num,
            "decomposition_cot": self.decomposition_cot
        }

        if self.is_valid:
            output = {
                **output,
                "op": self.op.value,
                "qpl": self.qpl,
                "prefix_qpl": self.prefix_qpl,
                "qpl_line": self.qpl_line,
                "children": [child.to_dict() for child in self.children],
            }

        return output


    @staticmethod
    def from_dict(tree_dict: Dict[str, Any]) -> "QPLQDTree":
        tree = QPLQDTree(
            question=tree_dict["question"],
            db_id=tree_dict["db_id"],
            op=Operator(tree_dict["op"]) if tree_dict['is_valid'] else None,  # type: ignore
            line_num=tree_dict["line_num"],
            qpl_line=tree_dict.get("qpl_line"),  # type: ignore
        )
        if tree_dict.get("children"):
            tree.children = tuple(QPLQDTree.from_dict(child) for child in tree_dict["children"])
            for child in tree.children:
                child.parent = tree
        return tree

    @property
    def prefix_qpl(self) -> str:
        if not self.children or any(child.qpl_line is None for child in self.children):
            return ""
        return "\n".join([(child.prefix_qpl + "\n" + child.qpl_line + f" ;  -- {child.question}").strip() for child in sorted(self.children, key=lambda x: x.line_num)])

    @property
    def qpl(self) -> str:
        if not self.qpl_line:
            return None
        output = ""
        if self.prefix_qpl:
            output += self.prefix_qpl + "\n"
        output += self.qpl_line + " ; -- " + self.question
        return output

    @property
    def is_valid(self) -> bool:
        return self.op is not None and all(child.is_valid for child in self.children)


@dataclass
class PartialQDTree:
    """A tree structure representing a qpl query with its partual decomposition - used for creating the supervised dataset."""
    question: str
    db_id: str

    parent: Optional["PartialQDTree"] = None
    op: Optional[Operator] = None
    children: Tuple["PartialQDTree", ...] = ()

    prefix_qpl: Optional[str] = None
    qpl_line: Optional[str] = None

    @property
    def idx(self) -> int:
        if not self.qpl_line:
            raise ValueError("QPL line is not set, cannot extract index.")
        m = re.match(r"#(\d+) = .*", self.qpl_line, re.DOTALL)
        if not m:
            raise ValueError(f"QPL line does not match expected format: {self.qpl_line}")
        return int(m.group(1))
    
    def is_complete(self) -> bool:
        """Check if the tree is complete, i.e., has a valid operator and QPL line."""
        return self.op is not None and self.qpl_line is not None and all(child.is_complete() for child in self.children)


    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "op": self.op.value if self.op else None,
            "children": [child.to_dict() for child in self.children],
            "prefix_qpl": self.prefix_qpl,
            "qpl_line": self.qpl_line
        }
    
