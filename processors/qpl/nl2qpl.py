import os
import json
from typing import List

import pydash
from transformers.trainer_callback import TrainerCallback

from ..base import BaseProcessor


def update_type(col_type):
    if "char" in col_type or col_type == "" or "text" in col_type or "var" in col_type:
        return "text"
    elif (
        "int" in col_type
        or "numeric" in col_type
        or "decimal" in col_type
        or "number" in col_type
        or "id" in col_type
        or "real" in col_type
        or "double" in col_type
        or "float" in col_type
    ):
        return "number"
    elif "date" in col_type or "time" in col_type:
        return "date"
    elif "boolean" in col_type or col_type == "bit":
        return "boolean"
    else:
        return "others"


class NL2QPLProcessor(BaseProcessor):
    def __init__(
        self, 
        **kwargs
    ):
        """
        Args:
            **kwargs: Additional arguments passed to the BaseProcessor.
        """
        # before_filter = len(dataset)
        # dataset = dataset.filter(lambda row: row["id"] in self._db_content)
        # after_filter = len(dataset)
        # print(f"Filtered {before_filter - after_filter} rows from the dataset.")
            
        super().__init__(**kwargs)

    
    @property
    def _db_content(self):
        try:
            return self.__db_content
        except AttributeError:
            parent_dir = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(parent_dir, "db_content.json")) as f:
                self.__db_content = json.load(f)
            return self.__db_content

    @property
    def _db_schemas(self):
        try:
            return self.__db_schemas
        except AttributeError:
            parent_dir = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(parent_dir, "db_schemas.json")) as f:
                self.__db_schemas = json.load(f)
            return self.__db_schemas    


    def _create_table_prompt(
        self, id, add_db_content=True, add_column_types=True, add_pk=True, add_fk=True
    ):
        db_id = self._db_content[id]["db_id"]
        tables = self._db_schemas[db_id]["tables"]
        pk = self._db_schemas[db_id].get("pk", None)
        fk = self._db_schemas[db_id].get("fk", None)

        content = self._db_content[id]["db_content"]

        formatted_columns = lambda table_name, columns: ",\n".join(
            [
                "\t{column_name}{column_type}{content}".format(
                    column_name=column[0],
                    column_type=f" {update_type(column[1])}" if add_column_types else "",
                    content=f" ( {' , '.join(content[table_name][column[0]])} )"
                    if add_db_content and pydash.has(content, f"{table_name}.{column[0]}")
                    else "",
                )
                for column in columns
            ]
        )

        formatted_table_pk = lambda table_pk: ",\n\tprimary key ( {table_pk} )".format(
            table_pk=" , ".join(table_pk)
        )

        formatted_table_fk = lambda table_fk: ",\n{table_fk}".format(
            table_fk=",\n".join(
                [
                    "\tforeign key ( {fk_columns_name} ) references {referenced_table_name} ( {referenced_columns_name} )".format(
                        fk_columns_name=" , ".join(
                            [fk_column[0] for fk_column in fk_columns]
                        ),
                        referenced_table_name=referenced_table_name,
                        referenced_columns_name=" , ".join(
                            [fk_column[1] for fk_column in fk_columns]
                        ),
                    )
                    for referenced_table_name, fk_columns in table_fk.items()
                ]
            )
        )

        prompt = "\n\n".join(
            [
                "CREATE TABLE {table_name} (\n{formatted_columns}{formatted_table_pk}{formatted_table_fk}\n)".format(
                    table_name=table_name,
                    formatted_columns=formatted_columns(table_name, columns),
                    formatted_table_pk=formatted_table_pk(pk[table_name])
                    if add_pk and pk and pydash.has(pk, table_name)
                    else "",
                    formatted_table_fk=formatted_table_fk(fk[table_name])
                    if add_fk and fk and pydash.has(fk, table_name)
                    else "",
                )
                for table_name, columns in tables.items()
            ]
        )

        return prompt


    def process_row(self, row):
        db_id = self._db_content[row['id']]["db_id"]

        prompt = (
            f"{db_id}\n\n"
            + self._create_table_prompt(row['id'])
            + "\n\n"
            + "-- Using valid QPL, answer the following questions for the tables provided above."
            + f"""\n\n-- {row["question"].strip()}\n\n[QPL]: """
        )

        return {"prompt": prompt, "response": f"{db_id} | {row['qpl']}"}
