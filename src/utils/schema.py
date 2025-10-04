import sys
import os
import json
import random
from enum import StrEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from math import sqrt

import src.utils.paths as p
from src.utils.lists import distribute_items


class NoiseStrategy(StrEnum):
    BREADTH = "breadth"  # distribute num_cols_to_add across as many tables as possible
    DEPTH = "depth"      # add num_cols_to_add to as few tables as possible
    MIXED = "mixed"


class SchemaRepresentation(StrEnum):
    DDL = "ddl"
    MARKDOWN = "markdown"


@dataclass
class PrimaryKey:
    col_name: str
    apply_lower: bool = False

    def __post_init__(self):
        if self.apply_lower:
            self.col_name = self.col_name.lower()
    

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
    
    def ddl(self):
        return f"FOREIGN KEY ({self.from_col}) REFERENCES {self.to_table.name}({self.to_col})"


@dataclass
class Column:
    name: str
    type: str
    is_pk: bool
    maps_to: str | None = None
    description: str | None = None
    examples_freq: dict[str, int] = field(default_factory=dict)
    constraint: Optional[str] = None
    apply_lower: bool = False
    simple_type: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    num_rows: int | None = None

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
        if "char" in t or t == "" or "text" in t or "var" in t:
            self.type = "text"
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
            self.type = "number"
        elif "date" in t or "time" in t:
            self.type = "date"
        elif "boolean" in t or t == "bit":
            self.type = "boolean"
        else:
            self.type = "others"

    def ddl(self):
        if self.constraint:
            return f"{self.name} {self.type} {self.constraint}"
        return f"{self.name} {self.type}"
    
    def metadata_desc(self, max_examples: int) -> str:
        if not self.metadata:
            return "N/A"
        
        output = f"This column has {self.metadata['num_nulls']} null values and {self.metadata['num_unique']} distinct values. "

        # values info
        values_general_info = ""
        if self.metadata['all_numeric']:
            values_general_info = "All of them are numeric"
        elif self.metadata['all_alphabetic']:
            values_general_info = "All of them are alphabetic"
            if self.metadata['all_lower']:
                values_general_info += " (lower case)"
            elif self.metadata['all_upper']:
                values_general_info += " (upper case)"
        elif self.metadata['all_alphanumeric']:
            values_general_info += "All of them are alpha-numeric"
        if values_general_info:
            output += values_general_info + ". "

        # distribution info
        if 'min_length' in self.metadata:
            if self.metadata['min_length'] == self.metadata['max_length']:
                output += f"The values are always {self.metadata['min_length']} characters long. "
            else:
                output += f"The values are between {self.metadata['min_length']} and {self.metadata['max_length']} characters long, with a mean of {self.metadata['mean_length']:.2f} and a standard deviation of {self.metadata['std_dev_length']:.2f}. "
        if 'min_value' in self.metadata:
            if self.metadata['min_value'] != self.metadata['max_value']:
                output += f"The values are between {self.metadata['min_value']} and {self.metadata['max_value']}, with a mean of {self.metadata['mean_value']:.2f} and a standard deviation of {self.metadata['std_dev']:.2f}. "
        if 'min_date' in self.metadata:
            output += f"The dates range from {self.metadata['min_date']} to {self.metadata['max_date']}. "

        # common prefixes and suffixes
        def wilson_lower_confidence_bound(freq: int) -> bool:
            # https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full
            if not self.num_rows:
                raise ValueError("num_rows must be set to calculate frequency thresholds.")
            z = 1.96
            tau = 0.05
            x = (freq/self.num_rows + z*z/(2*self.num_rows) - z * sqrt((freq/self.num_rows)*(1 - freq/self.num_rows)/self.num_rows + z*z/(4*self.num_rows*self.num_rows))) / (1 + z*z/self.num_rows)
            return x >= tau
        
        prefixes = {p:f for p,f in self.metadata['common_prefixes'].items() if wilson_lower_confidence_bound(f)}
        suffixes = {s:f for s,f in self.metadata['common_suffixes'].items() if wilson_lower_confidence_bound(f)}

        if prefixes and len(set(prefixes.keys()).intersection(set(self.examples_freq.keys()))) <= 1 and len(self.examples_freq) > 1:
            pref_str = ', '.join([f"{p} ({f} occurrences)" for p, f in prefixes.items()][:max_examples])
            output += f"Common prefixes: {pref_str}. "
        if suffixes and len(set(suffixes.keys()).intersection(set(self.examples_freq.keys()))) <= 1 and len(self.examples_freq) > 1:
            suff_str = ', '.join([f"{s} ({f} occurrences)" for s, f in suffixes.items()][:max_examples])
            output += f"Common suffixes: {suff_str}. "
        
        return output.strip()
    
    def markdown(self, examples_random_order: bool, max_examples: int) -> str:
        output = f"| {self.name} | {self.type.upper() if not self.apply_lower else self.type.lower()}"
        output += " | T" if self.is_pk else " | F"
        output += " | " + (self.maps_to if self.maps_to else "N/A")
        output += " | " + (self.description if self.description else "N/A")
        output += " | " + (self.metadata_desc(max_examples=max_examples) if self.metadata else "N/A")
        examples_str = "N/A"
        if self.examples_freq:
            examples_freq_items = list(self.examples_freq.items())[:max_examples]
            if examples_random_order:
                random.shuffle(examples_freq_items)
            examples_str = ', '.join([f"{e}" for e, f in examples_freq_items])  # optional: add frequency
        output += " | " + examples_str + " |"
        return output


class Table:
    name: str
    columns: List[Column]
    _pks: List[PrimaryKey]
    _fks: List[ForeignKey]
    num_rows: int | None = None

    def __init__(self, name: str, columns: List[Column], pks: List[PrimaryKey], fks: List[ForeignKey], apply_lower: bool = False, num_rows: int | None = None):
        self.name = name
        if apply_lower:
            self.name = self.name.lower()
        self.columns = columns
        self.pks = pks
        self.fks = fks
        self.apply_lower = apply_lower
        self.num_rows = num_rows

        assert all(col.apply_lower == apply_lower for col in columns), f"All columns must have the same case sensitivity as the table {self.name!r}."
        assert all(pk.apply_lower == apply_lower for pk in pks), f"All primary keys must have the same case sensitivity as the table {self.name!r}."
        assert all(fk.apply_lower == apply_lower for fk in fks), f"All foreign keys must have the same case sensitivity as the table {self.name!r}."

    @property
    def pks(self) -> List[PrimaryKey]:
        return self._pks
    
    @pks.setter
    def pks(self, pks: List[PrimaryKey]):
        assert all(pk.col_name in self.column_names for pk in pks), f"Primary keys {pks} must be a subset of columns of table {self.name!r}: {self.columns}"
        for column in self.columns:
            column.is_pk = False
            for pk in pks:
                if pk.col_name == column.name:
                    column.is_pk = True
        self._pks = pks
    
    @property
    def fks(self) -> List[ForeignKey]:
        return self._fks
    
    @fks.setter
    def fks(self, fks: List[ForeignKey]):
        assert all(fk.from_col in self.column_names for fk in fks), f"Foreign keys {fks} must be a subset of columns of table {self.name!r}: {self.columns}"
        for column in self.columns:
            for fk in fks:
                if fk.from_col == column.name:
                    column.maps_to = f"{fk.to_table.name}({fk.to_col})"
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
    
    def link(self, columns: set[str]) -> "Table":
        columns_lower = {col.lower() for col in columns}
        linked_table = Table(
            name=self.name,
            columns=[col for col in self.columns if col.is_pk or col.name.lower() in columns_lower],
            pks=self.pks,  # Always include all PKs
            fks=[fk for fk in self.fks if fk.from_col.lower() in columns_lower],
            apply_lower=self.apply_lower,
            num_rows=self.num_rows
        )

        # Update foreign keys reference to table instance
        for fk in linked_table.fks:
            fk.to_table = linked_table

        return linked_table

    def ddl(self):
        pk_str = ["PRIMARY KEY (" + ", ".join(pk.col_name for pk in self.pks) + ")"]
        cols_str = ",\n".join(f"    {col}" for col in [col.ddl() for col in self.columns] + pk_str + [fk.ddl() for fk in self.fks])
        return f"CREATE TABLE {self.name} (\n{cols_str}\n);"
    
    def markdown(self, examples_random_order: bool, max_examples: int) -> str:
        output = f"# Table: {self.name}"
        if self.num_rows is not None:
            output += f" ({self.num_rows} rows)"
        output += "\n## Columns"
        output += "\n| Name | Type | Is PK | Maps To | Description | Metadata | Examples |"
        output += "\n|---|---|---|---|---|---|---|\n"
        output += "\n".join([f"{col.markdown(examples_random_order=examples_random_order, max_examples=max_examples)}" for col in self.columns])
        return output
    
    def __len__(self):
        return len(self.columns)


class DBSchema:
    db_id: str
    tables: Dict[str, Table]

    def __init__(self, db_id: str, tables: Dict[str, Table], apply_lower: bool = False):
        self.db_id = db_id
        self.tables = tables if not apply_lower else {k.lower(): v for k,v in tables.items()}
        self.apply_lower = apply_lower
    
    @staticmethod
    def from_db_schemas_file(
        db_schemas_file: os.PathLike = p.DB_SCHEMAS_JSON_PATH,
        dbs_metadata_file: os.PathLike | None = p.DB_PROFILES_PATH, 
        columns_descriptions_file: os.PathLike | None = p.DB_COLS_DESCRIPTIONS_PATH,
        apply_lower: bool = False
    ) -> Dict[str, "DBSchema"]:
        with open(db_schemas_file, "r") as f:
            db_schemas = json.load(f)
        if dbs_metadata_file:
            with open(dbs_metadata_file, "r") as f:
                dbs_metadata = json.load(f)
        if columns_descriptions_file:
            with open(columns_descriptions_file, "r") as f:
                cols_descriptions = json.load(f)
        return DBSchema.from_db_schemas(db_schemas, dbs_metadata if dbs_metadata_file else None, cols_descriptions if columns_descriptions_file else None, apply_lower)

    @staticmethod
    def from_db_schemas(db_schemas: Dict, dbs_metadata: dict | None = None, cols_descriptions: dict | None = None, apply_lower: bool = False) -> Dict[str, "DBSchema"]:
        return {
            db_id: DBSchema.from_db_schema(db_id, tables_data, dbs_metadata[db_id] if dbs_metadata else None, cols_descriptions[db_id] if cols_descriptions else None, apply_lower)
            for db_id, tables_data in db_schemas.items()
        }
    
    @staticmethod
    def from_db_schema(db_id: str, db_schema: Dict, db_metadata: Dict[str, Any] | None, cols_descriptions: Dict[str, Any] | None, apply_lower: bool = False) -> "DBSchema":
        tables: Dict[str, Table] = {}
        
        for table_name, cols_data in db_schema['tables'].items():
            if table_name in tables:
                raise ValueError(f"Table {table_name!r} already exists in the schema.")
            
            tables[table_name] = Table(
                name=table_name,
                columns=[
                    Column(
                        name=col_name, 
                        type=col_type, 
                        is_pk=False, 
                        maps_to=None, 
                        description=cols_descriptions[table_name].get(col_name, None) if cols_descriptions else None,
                        constraint=col_constraint, 
                        apply_lower=apply_lower, 
                        examples_freq=db_metadata[table_name].get(col_name, {}).get('most_common_values', {}) if db_metadata else {},
                        metadata=db_metadata[table_name].get(col_name, {}) if db_metadata else {},
                        num_rows=db_metadata[table_name].get('num_rows', 0) if db_metadata else None,
                    ) for col_name, col_type, col_constraint in cols_data
                ],
                pks=[],
                fks=[],
                apply_lower=apply_lower,
                num_rows=db_metadata[table_name].get('num_rows', 0) if db_metadata else None,
            )

        for table_name, pk_cols in db_schema['pk'].items():
            tables[table_name].pks = [PrimaryKey(pk_col, apply_lower=apply_lower) for pk_col in pk_cols]
        
        for src_table_name, dst_tables_data in db_schema['fk'].items():
            src_table_fks = []
            src_table = tables[src_table_name]
            for dst_table_name, src_dst_fks in dst_tables_data.items():
                dst_table = tables[dst_table_name]
                for src_col, dst_col in src_dst_fks:
                    fk = ForeignKey(from_col=src_col, to_table=dst_table, to_col=dst_col, apply_lower=apply_lower)
                    src_table_fks.append(fk)
            src_table.fks = src_table_fks

        return DBSchema(db_id=db_id, tables=tables)
    
    @property
    def entities(self) -> List[str]:
        # primary keys that are not foreign keys to other tables distinguish the table as an entity
        return [
            table.name
            for table in self.tables.values()
            if set([pk.col_name for pk in table.pks]).difference([fk.from_col for fk in table.fks])
        ]
    
    def get_table(self, table_name: str) -> Table:
        if not self.apply_lower:
            table_name = table_name.lower()
        if table_name not in self.tables:
            raise KeyError(f"Table {table_name!r} does not exist in the schema {self.db_id!r}.")
        return self.tables[table_name]
    
    def _process_link_cols(self, table_cols: dict[str, set[str]], noise: float, noise_strategy: NoiseStrategy) -> dict[str, set[str]]:
        table_cols_lower = {
            table_name.lower(): {col_name.lower() for col_name in cols_names} 
            for table_name, cols_names in table_cols.items()
        }
        lower_to_original_tbname = {k.lower(): k for k in self.tables.keys()}

        # determine number of columns to add
        req_cols = sum(len(cols) for cols in table_cols.values())
        num_cols_to_add = round(noise * (len(self) - req_cols)) if 0 <= noise <= 1 else min(len(self) - req_cols, max(round(noise) - req_cols, 0))

        # map table names to the number of required columns, and the number of maximum columns
        table_to_req_cap = {
            table.name.lower(): (len(table_cols_lower.get(table.name.lower(), {})), len(table.columns))
            for table in self.tables.values()
        }
        
        if noise_strategy == NoiseStrategy.BREADTH or noise_strategy == NoiseStrategy.DEPTH:
            # Table selection
            if noise_strategy == NoiseStrategy.BREADTH:
                # take tables that are required, add tables with the most capacity
                selected_tables = [t for t in table_cols_lower]
                selected_tables += [
                    t for t, _ in sorted(
                        [(t, (req, cap)) for t, (req, cap) in table_to_req_cap.items() if t not in table_cols_lower],  # non-required tables
                        key=lambda x: x[1][1], reverse=True
                    )
                ][:num_cols_to_add]
            elif noise_strategy == NoiseStrategy.DEPTH:
                # take tables that are required, add tables with the most capacity until num_cols_to_add is reached
                selected_tables = [t for t in table_cols_lower]
                potential_extra_items = sum(len(self.tables[lower_to_original_tbname[t]].columns) - len(table_cols_lower[t]) for t in selected_tables)
                nonreq_tables = sorted(
                    [(t, (req, cap)) for t, (req, cap) in table_to_req_cap.items() if t not in table_cols_lower],  # non-required tables
                    key=lambda x: x[1][1], reverse=True
                )
                for t, _ in nonreq_tables:
                    if potential_extra_items >= num_cols_to_add:
                        break
                    selected_tables.append(t)
                    potential_extra_items += len(self.tables[lower_to_original_tbname[t]].columns)
                if potential_extra_items < num_cols_to_add:
                    raise ValueError(f"Not enough capacity to add {num_cols_to_add} columns. Max possible is {potential_extra_items}.")

            # Distribute columns to add across selected tables
            table_to_nsample = distribute_items(
                max_capacity={t: cap for t, (req, cap) in table_to_req_cap.items() if t in selected_tables},
                n_items=num_cols_to_add,
                init={t: len(table_cols_lower.get(t, set())) for t in selected_tables},
            )
            # adjust table_to_nsample to be the number of columns to add, not the total number of columns
            table_to_nsample = {t: n - len(table_cols_lower.get(t, set())) for t, n in table_to_nsample.items()}

            # Sample columns to add
            for table_name, n_sample in table_to_nsample.items():
                if n_sample > 0:
                    available_cols = set(
                        [col.lower() for col in self.tables[lower_to_original_tbname[table_name]].column_names]).difference(
                        table_cols_lower.get(table_name, set())
                    )
                    sampled_cols = random.sample(list(available_cols), n_sample)
                    if table_name in table_cols_lower:
                        table_cols_lower[table_name].update(sampled_cols)
                    else:
                        table_cols_lower[table_name] = set(sampled_cols)
        elif noise_strategy == NoiseStrategy.MIXED:
            # randomly choose between breadth and depth for each column addition
            flat_nonreq_items = [
                (table.name.lower(), col_name.lower()) 
                for table in self.tables.values() 
                for col_name in table.column_names 
                if col_name.lower() not in table_cols_lower.get(table.name.lower(), set())
            ]

            # Sample columns to add
            items_to_add = random.sample(flat_nonreq_items, num_cols_to_add)
            for table_name, col_name in items_to_add:
                if table_name in table_cols_lower:
                    table_cols_lower[table_name].add(col_name)
                else:
                    table_cols_lower[table_name] = {col_name}
        else:
            raise ValueError(f"Unknown noise strategy: {noise_strategy}")

        # Add primary keys of selected tables if not already included
        for table_name, selected_cols in table_cols_lower.items():
            table = self.tables[lower_to_original_tbname[table_name]]
            for pk in table.pks:
                selected_cols.add(pk.col_name.lower())

        return table_cols_lower
    
    
    def linked(self, table_cols: dict[str, set[str]], noise: float = 0, noise_strategy: NoiseStrategy = NoiseStrategy.MIXED) -> "DBSchema":
        table_cols_lower = self._process_link_cols(table_cols=table_cols, noise=noise, noise_strategy=noise_strategy)
        lower_to_original_tbname = {k.lower(): k for k in self.tables.keys()}
        linked_schema = DBSchema(
            db_id=self.db_id,
            tables={
                lower_to_original_tbname[tb_name]: self.tables[lower_to_original_tbname[tb_name]].link(columns) 
                for tb_name, columns in table_cols_lower.items()
            },
            apply_lower=self.apply_lower
        )

        # Ensure foreign keys are included only if both columns are requested
        for table in linked_schema.tables.values():
            table.fks = [
                fk for fk in table.fks
                if fk.to_col.lower() in table_cols_lower.get(fk.to_table.name.lower(), [])
            ]

        return linked_schema
    
    def ddl(self):
        tables_str = "\n\n".join(table.ddl() for table in self.tables.values())
        return f"Database Name: {self.db_id}\n```DDL\n{tables_str}\n```"
    
    def markdown(self, examples_random_order: bool = True, max_examples: int = 5) -> str:
        output = f"【DB_ID】 {self.db_id}\n"
        output += f"【Schema】\n"
        output += "\n".join([table.markdown(examples_random_order=examples_random_order, max_examples=max_examples) for table in self.tables.values()])
        output += "\n【Foreign Keys】\n"
        for table in self.tables.values():
            for fk in table.fks:
                output += f"{table.name}.{fk.from_col}->{fk.to_table.name}.{fk.to_col}\n"
        return output

    def __len__(self):
        return sum(len(t) for t in self.tables.values())


if __name__ == "__main__":
    db_schemas = DBSchema.from_db_schemas_file()

    schema = db_schemas[sys.argv[1]]
    representation = sys.argv[2]

    if representation == "ddl":
        print(schema.ddl())
    elif representation == "markdown":
        print(schema.markdown())
    else:
        raise ValueError(f"Unknown representation: {representation}. Use 'ddl' or 'markdown'.")