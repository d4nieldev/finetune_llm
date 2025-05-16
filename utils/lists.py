from typing import List, Tuple, TypeVar


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