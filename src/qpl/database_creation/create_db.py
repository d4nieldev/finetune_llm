import argparse
import json
import sqlite3
import re
from pathlib import Path
from typing import List
import urllib.parse

import numpy as np
import pandas as pd
import pyodbc
import urllib
import sqlalchemy
from sqlalchemy import create_engine
from tqdm.auto import tqdm

import utils.qpl.paths as p


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


def create_database(dbs: List[Path]):
    conn = pyodbc.connect(connection_string, autocommit=True)
    cursor = conn.cursor()
    for db in dbs:
        cursor.execute(f"""IF NOT EXISTS (
            SELECT * FROM sys.schemas WHERE name = '{db.stem}'
        )
        EXEC('CREATE SCHEMA [{db.stem}]');""")
    ddls = p.DB_CREATION_SCHEMAS_DDL_DIR.glob("**/*.sql")
    for ddl in ddls:
        sql_text = ddl.read_text().replace("USE spider;", "")
        # Split on GO (SQL Server batch separator)
        batches = [b.strip() for b in re.split(r'(?im)^[\s]*GO[\s]*$', sql_text) if b.strip()]
        for batch in batches:
            cursor.execute(batch)
    conn.close()


def get_tables(cursor):
    return [
        x[0].lower()
        for x in cursor.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]


def get_types(cursor, table):
    result = cursor.execute(f"SELECT * FROM pragma_table_info('{table}')").fetchall()
    return {x[1]: x[2] for x in result}


def convert_sqlite_type(schema, df, types):
    def try_(f):
        def g(x):
            if x == "inf":
                return np.nan
            try:
                return f(x)
            except:
                return np.nan

        return g

    for col, type_ in types.items():
        t = type_.lower()
        cl = col.lower()
        if t.startswith("int") or t.startswith("bigint") or "unsigned" in t:
            df[col] = df[col].apply(try_(int))
        elif (
            t.startswith("numeric")
            or t.startswith("float")
            or t.startswith("real")
            or t.startswith("double")
            or t.startswith("decimal")
        ):
            df[col] = df[col].apply(try_(float))
        elif schema == "car_1" and cl in ("horsepower", "mpg"):
            if cl == "horsepower":
                df[col] = df[col].apply(try_(int))
            else:
                df[col] = df[col].apply(try_(float))
        elif (
            schema == "student_transcripts_tracking"
            and cl == "transcript_date"
            and t == "datetime"
        ):
            df[col] = pd.to_datetime(df[col]).dt.year
        elif schema == "wta_1" and t == "date":
            df[col] = pd.to_datetime(df[col], format="%Y%m%d")
        else:  # keep other types as text
            pass

    return df


def dump(schema, conn, cursor):
    tables = get_tables(cursor)
    result = {}
    for t in tables:
        types = get_types(cursor, t)
        df = pd.read_sql(f"select * from {t}", conn)
        if schema == "orchestra" and t == "show":
            df.rename(
                {"If_first_show": "Result", "Result": "If_first_show"},
                axis=1,
                inplace=True,
            )
        result[t] = convert_sqlite_type(schema, df, types)
    return result


def dump_all(dbs: List[Path]):
    result = {}
    for db in dbs:
        conn = sqlite3.connect(db)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.cursor()
        schema = db.stem
        result[schema] = dump(schema, conn, cursor)
    return result


def fill_databases():
    conn = pyodbc.connect(connection_string, autocommit=True)
    cursor = conn.cursor()
    # No USE needed: connection_string already targets 'test'
    df = pd.read_pickle(p.DB_CREATION_PICKLE_DATA_PATH)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            cursor.execute(row["sql"], row["parameters"])
        except Exception:
            continue
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--spider-path", type=Path)
    args = parser.parse_args()
    spider_path: Path = args.spider_path

    dbs = list((spider_path / "database").glob("**/*.sqlite"))

    create_database(dbs)

    data = dump_all(dbs)

    params = urllib.parse.quote(connection_string)
    url = "mssql+pyodbc:///?odbc_connect={0}".format(params)
    engine = create_engine(
        url
    )
    sorted_tables_by_schema = json.load(open(p.DB_CREATION_TABLES_SORTED_PATH, "r"))
    for schema, table_data in (bar := tqdm(data.items())):
        bar.set_description(schema)
        for table_name in sorted_tables_by_schema[schema]:
            try:
                rows = table_data.get(table_name)
                if rows is not None:
                    rows.to_sql(
                        table_name,
                        engine,
                        schema=schema,
                        if_exists="append",
                        index=False,
                        chunksize=1,
                    )
            except sqlalchemy.exc.IntegrityError:
                pass

    engine.dispose()

    fill_databases()


if __name__ == "__main__":
    main()
