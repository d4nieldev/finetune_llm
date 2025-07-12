from itertools import chain, combinations
from typing import Dict, List, Tuple, Iterable, TypeVar, Iterator
import random

def flatten(list_of_lists: List[List]) -> Tuple[List, List[int]]:
    lengths = [len(sublist) for sublist in list_of_lists]
    flat_list = [item for sublist in list_of_lists for item in sublist]
    return flat_list, lengths


def unflatten(flat_list: List, lengths: List[int]) -> List[List]:
    result = []
    index = 0
    for length in lengths:
        result.append(flat_list[index:index + length])
        index += length
    return result


def split_train_test(dataset: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train and test sets."""
    split_index = int(len(dataset) * train_ratio)
    random.shuffle(dataset)
    return dataset[:split_index], dataset[split_index:]


T = TypeVar('T')
def powerset(iterable: Iterable[T], include_empty: bool = True) -> Iterator[Tuple[T, ...]]:
    "powerset([1,2,3]) --> [()] [(1,)] [(2,)] [(3,)] [(1, 2)] [(1, 3)] [(2, 3)] [(1, 2, 3)]"
    start = 0 if include_empty else 1
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(start, len(s)+1))


def find_sublist(haystack: list[T], needle: list[T]) -> int:
    """
    Return the first index i such that haystack[i:i+len(needle)] == needle.
    If needle is empty → 0            (matches at the very start)
    If no match      → -1
    """
    n, m = len(haystack), len(needle)
    if m == 0:
        return 0                      # empty list is a prefix by convention

    # slide a window the size of `needle` over `haystack`
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1