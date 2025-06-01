from typing import Dict, List, Tuple, TypeVar
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