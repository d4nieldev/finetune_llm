import json
from decimal import Decimal

import pyodbc
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date

from src.experiments.qpl.validate_qpl import execute_sql
from src.utilsema import DBSchema, Table
from src.utilsort paths as p


connection_string = (
    'Driver={ODBC Driver 18 for SQL Server};'
    'Server=tcp:spider-sql.database.windows.net,1433;'
    'Database=test;'
    'Uid=iloveqpl;'
    'Pwd=P4$$w0rd!;'
    'Encrypt=yes;'
    'TrustServerCertificate=no;'
    'Connection Timeout=30;'
)


def convert_to_python(obj):
    if isinstance(obj, dict):
        return {
            str(convert_to_python(k)): convert_to_python(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, date):
        return str(obj)
    else:
        return obj


def profile_table(db_id: str, table: Table, cursor: pyodbc.Cursor, prefix_length: int = 3, suffix_length: int = 3, k: int = 10) -> dict[str, str]:
    table_data = execute_sql(cursor, f"SELECT * FROM [{db_id}].[{table.name}]")
    if not table_data:
        return {}

    df = pd.DataFrame(table_data)

    table_metadata = {}
    for col in table.columns:
        colname = col.name
        shape_info = {}
        if col.type == "text":
            shape_info = {
                "min_length": df[colname].apply(lambda x: len(x) if isinstance(x, str) else 0).min(),
                "max_length": df[colname].apply(lambda x: len(x) if isinstance(x, str) else 0).max(),
                "mean_length": df[colname].apply(lambda x: len(x) if isinstance(x, str) else 0).mean(),
                "std_dev_length": df[colname].apply(lambda x: len(x) if isinstance(x, str) else 0).std(),
            }
        elif col.type == "number":
            df[colname] = df[colname].apply(lambda x: np.nan if isinstance(x, str) and x == '' else x)
            shape_info = {
                "min_value": df[colname].astype(float).min(),
                "max_value": df[colname].astype(float).max(),
                "mean_value": df[colname].astype(float).mean(),
                "std_dev": df[colname].astype(float).std(),
            }
        elif col.type == "date":
            shape_info = {
                "min_date": pd.to_datetime(df[colname], errors='coerce').min().date(),
                "max_date": pd.to_datetime(df[colname], errors='coerce').max().date(),
            }
        table_metadata[colname] = {
            "num_nulls": df[colname].isnull().sum(),
            "num_unique": df[colname].nunique(),
            **shape_info,
            "all_lower": df[colname].apply(lambda x: isinstance(x, str) and x.islower()).all(),
            "all_upper": df[colname].apply(lambda x: isinstance(x, str) and x.isupper()).all(),
            "all_numeric": df[colname].apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.isnumeric())).all(),
            "all_alphabetic": df[colname].apply(lambda x: isinstance(x, str) and x.isalpha()).all(),
            "all_alphanumeric": df[colname].apply(lambda x: isinstance(x, str) and x.isalnum()).all(),
            "common_prefixes": df[colname].apply(lambda x: x[:prefix_length] if isinstance(x, str) else None).value_counts().head(k).to_dict(),
            "common_suffixes": df[colname].apply(lambda x: x[-suffix_length:] if isinstance(x, str) else None).value_counts().head(k).to_dict(),
            "most_common_values": df[colname].value_counts().head(k).to_dict(),
        }
    table_metadata["num_rows"] = len(df)

    return table_metadata


def profile_schema(db_schema: DBSchema, cursor: pyodbc.Cursor) -> dict[str, dict[str, str]]:
    table_to_cols_profiles = {}
    for table_name, table in db_schema.tables.items():
        table_to_cols_profiles[table_name] = profile_table(db_schema.db_id, table, cursor)
    return table_to_cols_profiles


def main():
    conn = pyodbc.connect(connection_string, autocommit=True)
    cursor = conn.cursor()
    db_schemas = DBSchema.from_db_schemas_file(dbs_metadata_file=None)
    schema_to_table_to_cols_profiles = {}
    for db_id, db_schema in tqdm(db_schemas.items(), desc="Profiling databases"):
        schema_to_table_to_cols_profiles[db_id] = profile_schema(db_schema, cursor)

    schema_to_table_to_cols_profiles = convert_to_python(schema_to_table_to_cols_profiles)
    with open(p.DB_PROFILES_PATH, 'w') as f:
        json.dump(schema_to_table_to_cols_profiles, f, indent=2)


if __name__ == "__main__":
    main()