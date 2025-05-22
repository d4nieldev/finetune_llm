import os
import json
import pydash
from typing import Any, Mapping
from abc import abstractmethod
import logging

from processors.base import BaseProcessor


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


class QPLProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load databases
        parent_dir = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(parent_dir, "db_content.json")) as f:
            self._db_content = json.load(f)
        
        with open(os.path.join(parent_dir, "db_schemas.json")) as f:
            self._db_schemas = json.load(f)
        
        self.__log_cache = set()

    
    def _create_table_prompt(
        self, 
        example: Mapping[str, Any],
        add_db_content=True,
        add_column_types=True,
        add_pk=True,
        add_fk=True,
        log_when_parent_not_found=True,
    ):
        try:
            example_id = self._example_to_id(example)
        except ValueError as e:
            if log_when_parent_not_found and str(e) not in self.__log_cache:
                logging.warning("%s", e)
            self.__log_cache.add(str(e))
            db_id = example['db_id']
            add_db_content = False
        else:
            db_id = self._db_content[example_id]["db_id"]

        # problematic ids
        if db_id == "car_11":
            db_id = "car_1"

        tables = self._db_schemas[db_id]["tables"]
        pk = self._db_schemas[db_id].get("pk", None)
        fk = self._db_schemas[db_id].get("fk", None)

        if add_db_content:
            content = self._db_content[example_id]["db_content"]

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
    
    @abstractmethod
    def _example_to_id(self, example: Mapping[str, Any]) -> str:
        """
        Convert an example to its corresponding ID for extracting its db content.

        Args:
            example (Mapping[str, Any]): The example to convert.

        Returns:
            str: The ID of the example.
        """
        raise NotImplementedError


