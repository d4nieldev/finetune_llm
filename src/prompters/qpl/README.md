# QPL Prompters

This folder contains prompters for QPL datasets.

All QPL prompters extend [QPLPrompter](base.py), which contains the function `_create_table_prompt()` that is responsible for generating the (rich) table schema for a given row in a qpl dataset.

In addition to the definitions for a general prompter, a QPL prompter must also define the function `_example_to_id()` that recieves an example and tries to get the `id` of that example in `db_content.json`. If the `id` cannot be retrieved, this function should raise a `ValueError`.

## Additional Files

* [`db_schemas.json`](db_schemas.json) - information about all the possible databases to generate the table schema.
* [`db_content.json`](db_content.json) - additional enrichment information for each question in the training/testing set (not relevant for inference)
