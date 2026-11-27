import io
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

ETL_DIR = Path(__file__).resolve().parent
CSV_PATH = ETL_DIR.parent / "ML" / "smartgrid_5yr_hourly_natural-selected.csv"
SQL_DIR = ETL_DIR.parent / "SQL"
OUTPUT_DIR = ETL_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "smart_grid_db"),
    "user": os.getenv("PGUSER", os.getenv("USER", "postgres")),
    "password": os.getenv("PGPASSWORD", ""),
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432)),
}

NUM_METERS = 100


def extract_and_clean(path: Path):
    df = pd.read_csv(path)
    raw_rows = len(df)

    # 1. Quality checks
    nulls = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())
    neg_loads = int((df["load_kw"] < 0).sum())

    # 2. Clean & Transform
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date

    expected_hours = int((df["timestamp"].max() - df["timestamp"].min()).total_seconds() // 3600) + 1
    missing_hours = expected_hours - len(df)

    if "is_lockdown" in df.columns:
        df = df.drop(columns=["is_lockdown"])

    # Collapse season columns
    df["season"] = "harmattan"
    df.loc[df["season_dry"], "season"] = "dry"
    df.loc[df["season_rainy"], "season"] = "rainy"
    df = df.drop(columns=["season_dry", "season_rainy", "season_harmattan"])

    # Collapse holiday_type columns
    df["holiday_type"] = "none"
    df.loc[df["holiday_type_national"] == 1, "holiday_type"] = "national"
    df.loc[df["holiday_type_religious"] == 1, "holiday_type"] = "religious"
    df = df.drop(columns=["holiday_type_national", "holiday_type_religious"])

    # 3. Save quality report
    report = pd.DataFrame([
        {"metric": "raw_rows", "value": raw_rows},
        {"metric": "clean_rows", "value": len(df)},
        {"metric": "null_values", "value": nulls},
        {"metric": "duplicate_rows", "value": dups},
        {"metric": "negative_load_values", "value": neg_loads},
        {"metric": "missing_hourly_timestamps", "value": missing_hours},
        {"metric": "dropped_columns", "value": "is_lockdown"},
    ])
    report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)
    print(report.to_string(index=False))
    return df


def build_dim_date(df: pd.DataFrame):
    # One row per day; daily average for temperature
    return (
        df.groupby("date", as_index=False)
        .agg({
            "weekday": "first",
            "is_weekend": "first",
            "month": "first",
            "season": "first",
            "temperature": "mean",
            "is_rain_day": "max",
            "holiday_type": "first",
        })
        .assign(temperature=lambda d: d["temperature"].round(2))
    )


def build_dim_meter(num_meters=100, seed=42):
    rng = np.random.default_rng(seed)
    n_res = int(num_meters * 0.60)
    n_com = int(num_meters * 0.25)
    n_ind = num_meters - n_res - n_com
    sectors = ["residential"] * n_res + ["commercial"] * n_com + ["industrial"] * n_ind

    weights = {"residential": 1.0, "commercial": 3.5, "industrial": 8.0}
    raw_weights = np.array([weights[s] * rng.uniform(0.8, 1.2) for s in sectors])
    shares = raw_weights / raw_weights.sum()

    return pd.DataFrame({
        "meter_id": [f"MTR_{i+1:04d}" for i in range(num_meters)],
        "consumer_type": sectors,
        "load_share": np.round(shares, 6),
    })


def create_schema(conn):
    with conn.cursor() as cur:
        cur.execute("""
            DROP MATERIALIZED VIEW IF EXISTS mv_load_denorm;
            DROP VIEW IF EXISTS v_load_denorm;
            DROP TABLE IF EXISTS fact_load;
            DROP TABLE IF EXISTS dim_date;
            DROP TABLE IF EXISTS dim_meter;

            CREATE TABLE dim_date (
                date DATE PRIMARY KEY,
                weekday SMALLINT NOT NULL,
                is_weekend SMALLINT NOT NULL,
                month SMALLINT NOT NULL,
                season VARCHAR(20) NOT NULL,
                temperature DOUBLE PRECISION NOT NULL,
                is_rain_day SMALLINT NOT NULL,
                holiday_type VARCHAR(20) NOT NULL
            );

            CREATE TABLE dim_meter (
                meter_id VARCHAR(16) PRIMARY KEY,
                consumer_type VARCHAR(20) NOT NULL,
                load_share DOUBLE PRECISION NOT NULL
            );

            CREATE TABLE fact_load (
                meter_id VARCHAR(16) REFERENCES dim_meter(meter_id),
                timestamp TIMESTAMP NOT NULL,
                date DATE REFERENCES dim_date(date),
                hour SMALLINT NOT NULL,
                load_kw DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (meter_id, timestamp)
            );
        """)
    conn.commit()


def copy_df(conn, df: pd.DataFrame, table_name: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table_name} FROM STDIN WITH (FORMAT CSV)", buf)
    conn.commit()

def create_views(conn):
    with conn.cursor() as cur:
        cur.execute((SQL_DIR / "denormalize.sql").read_text())
    conn.commit()

def main():
    t0 = time.perf_counter()
    clean_df = extract_and_clean(CSV_PATH)
    dim_date_df = build_dim_date(clean_df)
    dim_meter_df = build_dim_meter(NUM_METERS)

    conn = psycopg2.connect(**DB_CONFIG)
    create_schema(conn)

    # 1. Load dimensions
    copy_df(conn, dim_date_df, "dim_date")
    copy_df(conn, dim_meter_df, "dim_meter")
    print(f"\nLoaded dim_date ({len(dim_date_df):,} rows) and dim_meter ({len(dim_meter_df):,} rows)")

    # 2. Disaggregate and stream-load fact_load
    rng = np.random.default_rng(42)
    base_ts = clean_df["timestamp"].values
    base_date = clean_df["date"].values
    base_hour = clean_df["hour"].values
    base_load = clean_df["load_kw"].values

    total_rows = 0
    for _, row in dim_meter_df.iterrows():
        noise = rng.normal(1.0, 0.05, size=len(base_load))
        meter_load = np.clip(base_load * row["load_share"] * noise, 0.0, None)
        batch_df = pd.DataFrame({
            "meter_id": row["meter_id"],
            "timestamp": base_ts,
            "date": base_date,
            "hour": base_hour,
            "load_kw": np.round(meter_load, 4),
        })
        copy_df(conn, batch_df, "fact_load")
        total_rows += len(batch_df)

    elapsed = time.perf_counter() - t0
    print(f"Loaded fact_load ({total_rows:,} rows) in {elapsed:.2f}s ({total_rows / elapsed:,.0f} rows/sec)")

    # 3. Build the denormalised views for the ML data loader
    create_views(conn)
    print("Built v_load_denorm and mv_load_denorm")
    conn.close()


if __name__ == "__main__":
    main()
