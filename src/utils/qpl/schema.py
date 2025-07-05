import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import src.utils.qpl.paths as p


@dataclass
class PrimaryKey:
    col_name: str
    apply_lower: bool = False

    def __post_init__(self):
        if self.apply_lower:
            self.col_name = self.col_name.lower()

    def __str__(self):
        return f"{self.col_name}"
    

@dataclass
class ForeignKey:
    from_col: str
    to_table: "Table"
    to_col: str
    apply_lower: bool = False

    def __post_init__(self):
        if self.apply_lower:
            self.from_col = self.from_col.lower()
            self.to_col = self.to_col.lower()

        assert self.to_col in self.to_table.column_names, f"Foreign key column {self.to_col!r} does not exist in table {self.to_table.name!r}"
        assert self.apply_lower == self.to_table.apply_lower, f"Foreign key case sensitivity ({self.apply_lower}) does not match table {self.to_table.name!r} case sensitivity ({self.to_table.apply_lower})"

    def __repr__(self):
        return f"ForeignKey( {self.from_col!r} -> {self.to_table.name!r}.{self.to_col!r} )"
    
    def __str__(self):
        return f"FOREIGN KEY ({self.from_col}) REFERENCES {self.to_table.name}({self.to_col})"
    

@dataclass
class Column:
    name: str
    type: str
    constraint: Optional[str] = None
    apply_lower: bool = False
    simple_type: bool = True

    def __post_init__(self):
        if self.apply_lower:
            self.name = self.name.lower()
            self.type = self.type.lower()
            if self.constraint:
                self.constraint = self.constraint.lower()
        if self.simple_type:
            self._simplify_type()

    def _simplify_type(self):
        t = self.type.lower()
        if "char" in t or self.type == "" or "text" in t or "var" in t:
            return "text"
        elif (
            "int" in t
            or "numeric" in t
            or "decimal" in t
            or "number" in t
            or "id" in t
            or "real" in t
            or "double" in t
            or "float" in t
        ):
            return "number"
        elif "date" in t or "time" in t:
            return "date"
        elif "boolean" in t or t == "bit":
            return "boolean"
        else:
            return "others"

    def __str__(self):
        if self.constraint:
            return f"{self.name} {self.type} {self.constraint}"
        return f"{self.name} {self.type}"


class Table:
    name: str
    columns: List[Column]
    _pks: List[PrimaryKey]
    _fks: List[ForeignKey]

    def __init__(self, name: str, columns: List[Column], pks: List[PrimaryKey], fks: List[ForeignKey], apply_lower: bool = False):
        self.name = name
        if apply_lower:
            self.name = self.name.lower()
        self.columns = columns
        self.pks = pks
        self.fks = fks
        self.apply_lower = apply_lower

        assert all(col.apply_lower == apply_lower for col in columns), f"All columns must have the same case sensitivity as the table {self.name!r}."
        assert all(pk.apply_lower == apply_lower for pk in pks), f"All primary keys must have the same case sensitivity as the table {self.name!r}."
        assert all(fk.apply_lower == apply_lower for fk in fks), f"All foreign keys must have the same case sensitivity as the table {self.name!r}."

    @property
    def pks(self) -> List[PrimaryKey]:
        return self._pks
    
    @pks.setter
    def pks(self, pks: List[PrimaryKey]):
        assert all(pk.col_name in self.column_names for pk in pks), f"Primary keys {pks} must be a subset of columns of table {self.name!r}: {self.columns}"
        self._pks = pks
    
    @property
    def fks(self) -> List[ForeignKey]:
        return self._fks
    
    @fks.setter
    def fks(self, fks: List[ForeignKey]):
        assert all(fk.from_col in self.column_names for fk in fks), f"Foreign keys {fks} must be a subset of columns of table {self.name!r}: {self.columns}"
        self._fks = fks

    @property
    def column_names(self) -> List[str]:
        return [col.name for col in self.columns]
    
    def src_colname_table(self, colname: str) -> Tuple[str, 'Table']:
        if not self.apply_lower:
            colname = colname.lower()
        if fk := next((fk for fk in self.fks if fk.from_col == colname and fk.to_table != self), None):
            return fk.to_table.src_colname_table(fk.from_col)
        return colname, self
    
    def __str__(self):
        pk_str = ["PRIMARY KEY (" + ", ".join(pk.col_name for pk in self.pks) + ")"]
        cols_str = ",\n".join(f"    {col}" for col in self.columns + pk_str + self.fks)
        return f"CREATE TABLE {self.name} (\n{cols_str}\n);"


class DBSchema:
    db_id: str
    _tables: Dict[str, Table]

    def __init__(self, db_id: str, tables: Dict[str, Table], apply_lower: bool = False):
        self.db_id = db_id
        self._tables = tables if apply_lower else {k.lower(): v for k,v in tables.items()}
        self.apply_lower = apply_lower
    
    @staticmethod
    def from_db_schemas_file(db_schemas_file: os.PathLike, apply_lower: bool = False) -> Dict[str, "DBSchema"]:
        with open(db_schemas_file, "r") as f:
            db_schemas = json.load(f)
        return DBSchema.from_db_schemas(db_schemas, apply_lower)
    
    @staticmethod
    def from_db_schemas(db_schemas: Dict, apply_lower: bool = False) -> Dict[str, "DBSchema"]:
        return {
            db_id: DBSchema.from_db_schema(db_id, tables_data, apply_lower) 
            for db_id, tables_data in db_schemas.items()
        }
    
    @staticmethod
    def from_db_schema(db_id: str, db_schema: Dict, apply_lower: bool = False) -> "DBSchema":
        tables: Dict[str, Table] = {}
        
        for table_name, cols_data in db_schema['tables'].items():
            if table_name in tables:
                raise ValueError(f"Table {table_name!r} already exists in the schema.")
            
            tables[table_name] = Table(
                name=table_name,
                columns=[
                    Column(name=col_name, type=col_type, constraint=col_constraint, apply_lower=apply_lower) 
                    for col_name, col_type, col_constraint in cols_data
                ],
                pks=[],
                fks=[],
                apply_lower=apply_lower
            )

        for table_name, pk_cols in db_schema['pk'].items():
            tables[table_name].pks = [PrimaryKey(pk_col, apply_lower=apply_lower) for pk_col in pk_cols]
        
        for src_table_name, dst_tables_data in db_schema['fk'].items():
            src_table = tables[src_table_name]
            for dst_table_name, src_dst_fks in dst_tables_data.items():
                dst_table = tables[dst_table_name]
                for src_col, dst_col in src_dst_fks:
                    fk = ForeignKey(from_col=src_col, to_table=dst_table, to_col=dst_col, apply_lower=apply_lower)
                    src_table.fks.append(fk)

        return DBSchema(db_id=db_id, tables=tables)
    
    @property
    def entities(self) -> List[str]:
        # primary keys that are not foreign keys to other tables distinguish the table as an entity
        return [
            table.name
            for table in self._tables.values()
            if set([pk.col_name for pk in table.pks]).difference([fk.from_col for fk in table.fks])
        ]
    
    def get_table(self, table_name: str) -> Table:
        if not self.apply_lower:
            table_name = table_name.lower()
        if table_name not in self._tables:
            raise KeyError(f"Table {table_name!r} does not exist in the schema {self.db_id!r}.")
        return self._tables[table_name]
    
    def __str__(self):
        tables_str = "\n\n".join(str(table) for table in self._tables.values())
        return f"```DDL\n{tables_str}\n```"


if __name__ == "__main__":
    db_schemas = DBSchema.from_db_schemas_file(p.DB_SCHEMAS_JSON_PATH, apply_lower=False)
    schema = db_schemas["battle_death"]
    print(schema)