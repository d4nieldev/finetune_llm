import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from inference.qpl.types.qpl_types import Entity
import utils.qpl.paths as p


@dataclass
class PrimaryKey:
    col_name: str

    def __str__(self):
        return f"{self.col_name}"
    

@dataclass
class ForeignKey:
    from_col: str
    to_table: "Table"
    to_col: str

    def __post_init__(self):
        assert self.to_col in self.to_table.columns_names, f"Foreign key column {self.to_col!r} does not exist in table {self.to_table.name!r}"

    def __repr__(self):
        return f"ForeignKey( {self.from_col!r} -> {self.to_table.name!r}.{self.to_col!r} )"
    
    def __str__(self):
        return f"FOREIGN KEY ({self.from_col}) REFERENCES {self.to_table.name}({self.to_col})"
    

@dataclass
class Column:
    name: str
    type: str
    constraint: Optional[str] = None

    def __str__(self):
        if self.constraint:
            return f"{self.name} {self.type} {self.constraint}"
        return f"{self.name} {self.type}"


class Table:
    name: str
    columns: List[Column]
    _pks: List[PrimaryKey]
    _fks: List[ForeignKey]

    def __init__(self, name: str, columns: List[Column], pks: List[PrimaryKey], fks: List[ForeignKey]):
        self.name = name
        self.columns = columns
        self.pks = pks
        self.fks = fks

    @property
    def pks(self) -> List[PrimaryKey]:
        return self._pks
    
    @pks.setter
    def pks(self, pks: List[PrimaryKey]):
        assert all(pk.col_name in self.columns_names for pk in pks), f"Primary keys {pks} must be a subset of columns of table {self.name!r}: {self.columns}"
        self._pks = pks
    
    @property
    def fks(self) -> List[ForeignKey]:
        return self._fks
    
    @fks.setter
    def fks(self, fks: List[ForeignKey]):
        assert all(fk.from_col in self.columns_names for fk in fks), f"Foreign keys {fks} must be a subset of columns of table {self.name!r}: {self.columns}"
        self._fks = fks

    @property
    def columns_names(self) -> List[str]:
        return [col.name for col in self.columns]
    
    def __str__(self):
        pk_str = ["PRIMARY KEY (" + ", ".join(pk.col_name for pk in self.pks) + ")"]
        cols_str = ",\n".join(f"    {col}" for col in self.columns + pk_str + self.fks)
        return f"CREATE TABLE {self.name} (\n{cols_str}\n);"


class DBSchema:
    db_id: str
    tables: Dict[str, Table]

    def __init__(self, db_id: str, tables: Dict[str, Table]):
        self.db_id = db_id
        self.tables = tables

    @property
    def entities(self) -> List[Entity]:
        # primary keys that are not foreign keys to other tables distinguish the table as an entity
        return [
            Entity(table.name)
            for table in self.tables.values()
            if set([pk.col_name for pk in table.pks]).difference([fk.from_col for fk in table.fks])
        ]
    
    def __getitem__(self, item: str) -> Table:
        return self.tables[item]
    
    @staticmethod
    def from_db_schemas_file(db_schemas_file: os.PathLike) -> Dict[str, "DBSchema"]:
        with open(db_schemas_file, "r") as f:
            db_schemas = json.load(f)
        return DBSchema.from_db_schemas(db_schemas)
    
    @staticmethod
    def from_db_schemas(db_schemas: Dict) -> Dict[str, "DBSchema"]:
        return {
            db_id: DBSchema.from_db_schema(db_id, tables_data) 
            for db_id, tables_data in db_schemas.items()
        }
    
    @staticmethod
    def from_db_schema(db_id: str, db_schema: Dict) -> "DBSchema":
        tables: Dict[str, Table] = {}
        
        for table_name, cols_data in db_schema['tables'].items():
            if table_name in tables:
                raise ValueError(f"Table {table_name!r} already exists in the schema.")
            
            tables[table_name] = Table(
                name=table_name,
                columns=[
                    Column(name=col_name, type=col_type, constraint=col_constraint) 
                    for col_name, col_type, col_constraint in cols_data
                ],
                pks=[],
                fks=[]
            )

        for table_name, pk_cols in db_schema['pk'].items():
            tables[table_name].pks = [PrimaryKey(pk_col) for pk_col in pk_cols]
        
        for src_table_name, dst_tables_data in db_schema['fk'].items():
            src_table = tables[src_table_name]
            for dst_table_name, src_dst_fks in dst_tables_data.items():
                dst_table = tables[dst_table_name]
                for src_col, dst_col in src_dst_fks:
                    fk = ForeignKey(from_col=src_col, to_table=dst_table, to_col=dst_col)
                    src_table.fks.append(fk)

        return DBSchema(db_id=db_id, tables=tables)
    
    def __str__(self):
        tables_str = "\n\n".join(str(table) for table in self.tables.values())
        return f"```DDL\n{tables_str}\n```"


if __name__ == "__main__":
    db_schemas = DBSchema.from_db_schemas_file(p.DB_SCHEMAS_JSON_PATH)
    schema = db_schemas["battle_death"]
    print(schema)
    print()
    print(schema.entities)