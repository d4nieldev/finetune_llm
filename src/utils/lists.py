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


def distribute_items(max_capacity: dict[str, int], n_items: int, init: dict[str, int] = {}) -> dict[str, int]:
    bins = init.copy()
    remaining_items = n_items
    
    while remaining_items > 0:
        # Get bins that have space and their current counts
        available_bins = []
        for bin_name in bins:
            space = max_capacity[bin_name] - bins[bin_name]
            if space > 0:
                available_bins.append((bin_name, bins[bin_name], space))
        
        if not available_bins:
            raise ValueError(f"Not enough capacity to distribute all items. Bins: {bins}, Max Capacity: {max_capacity}, Remaining Items: {remaining_items}")
        
        # Sort by current count (ascending) to fill emptier bins first
        available_bins.sort(key=lambda x: x[1])
        
        # Calculate how many items each bin should get in this round
        min_count = available_bins[0][1]  # Current count of emptiest bin
        
        # Find how many bins are at the minimum level
        bins_at_min = sum(1 for _, count, _ in available_bins if count == min_count)
        
        # Distribute items to bring all minimum bins up one level
        items_this_round = min(bins_at_min, remaining_items)
        
        for bin_name, current_count, space in available_bins:
            if current_count == min_count and items_this_round > 0 and space > 0:
                bins[bin_name] += 1
                remaining_items -= 1
                items_this_round -= 1
    
    return bins