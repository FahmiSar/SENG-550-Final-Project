"""
Notes for Fahmi

This file will be used to take data from whatever source we have and convert it to a pandas df.
If you *reeeeeally* want to be fancy, you can also denormalise the tables that we get so that 
models.py can just become a plug and play. As for how you want to denormalise, up to you. I can adjust 
models accordingly if I have to.


Leave the data_loader_csv file as it is, that isn't gonna be that important anyway unless you reckon
we can include that in the report to show what that simple form of storage would look like (which 
is a shout tbh). Otherwise I think it's just postgres and denormalisation? 

Just don't change the data, only change (if you think it is a good idea) how it is stored in the df.
"""


import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_DIR / ".env", override=True)

DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "smart_grid_db"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432)),
}


# --------------------------------------------------
# PostgreSQL Data Loader
# --------------------------------------------------

def data_loader_postgres(db_config=None):
    """
    Retrieves the denormalized materialized view from PostgreSQL
    and converts it into a Pandas DataFrame.

    Returns:
        pd.DataFrame: Denormalized electricity consumption data.
    """

    if db_config is None:
        db_config = DB_CONFIG

    query = """
        SELECT *
        FROM mv_load_denorm
        ORDER BY meter_id, timestamp;
    """

    conn = None

    try:
        conn = psycopg2.connect(**db_config)

        df = pd.read_sql_query(query, conn)

        # Normalize data types
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = pd.to_datetime(df["date"])

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        print("Successfully loaded denormalized PostgreSQL data.")
        print(f"Rows loaded: {len(df):,}")
        print(f"Columns loaded: {len(df.columns)}")

        return df

    finally:
        if conn is not None:
            conn.close()


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    df = data_loader_postgres()

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset information:")
    df.info()

