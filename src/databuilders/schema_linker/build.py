import json
from collections import defaultdict

from datasets import load_dataset, DatasetDict, Dataset

from src.utils.tree import QPLTree


def create_dataset() -> DatasetDict:
    completer_ds = load_dataset("d4nieldev/qpl-completer-ds")
    schema_linker_ds = {}

    for split in completer_ds:
        ds_dict = defaultdict(list)
        for row in completer_ds[split]:
            total_qpl = ""
            if row['prefix_qpl']:
                total_qpl += row['prefix_qpl'] + "\n"
            total_qpl += row['qpl_line'] + " ; --" + row['question']
            total_qpl = total_qpl.replace(" Top ", " TopSort ")

            qpl_lines = [line.split(';')[0].strip() for line in total_qpl.split("\n") if line.strip()]
            qpl_tree = QPLTree.from_qpl_lines(qpl_lines)
            schema_items = qpl_tree.get_schema_items()

            ds_dict['db_id'].append(row['db_id'])
            ds_dict['question'].append(row['question'])
            ds_dict['qpl'].append(total_qpl)
            ds_dict['schema_items'].append(json.dumps({table: list(cols) for table, cols in schema_items.items()}))
        schema_linker_ds[split] = Dataset.from_dict(ds_dict)

    return DatasetDict(schema_linker_ds)


if __name__ == "__main__":
    dataset = create_dataset()
    dataset.push_to_hub("d4nieldev/qpl-schema-linker-ds")