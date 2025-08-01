import json
from typing import Any, Mapping, Literal
from abc import abstractmethod

from src.prompters.base import BasePrompter
import src.utils.qpl.paths as p
from src.utils.qpl.schema import DBSchema


class QPLPrompter(BasePrompter):
    def __init__(self, schema_representation: Literal["ddl", "m_schema"] = 'm_schema', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.schema_representation = schema_representation
        self.__db_schemas = DBSchema.from_db_schemas_file(p.DB_SCHEMAS_JSON_PATH, apply_lower=False)

    
    def tokens_to_add(self) -> list[str]:
        if self.schema_representation == "m_schema":
            return ["【", "】"]
        return super().tokens_to_add()

    
    def _get_schema_str(
        self, 
        db_id: str,
    ):
        # problematic ids
        if db_id == "car_11":
            db_id = "car_1"

        db_schema = self.__db_schemas[db_id]

        if self.schema_representation == "ddl":
            return db_schema.ddl()
        elif self.schema_representation == "m_schema":
            return db_schema.m_schema()
        else:
            raise ValueError(f"Unknown representation: {self.schema_representation}. Use 'ddl' or 'm_schema'.")
