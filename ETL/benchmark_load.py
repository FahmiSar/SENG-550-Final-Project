import io
import time
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from etl import CSV_PATH, DB_CONFIG, OUTPUT_DIR, build_dim_meter, extract_and_clean

BENCH_TABLE = "bench_fact_load"
FULL_ROWS = 4_384_800  # 43,848 hours x 100 meters


def build_sample_batch():
    """Build 1 meter's worth of fact_load rows (43,848 rows) for benchmarking."""
    clean_df = extract_and_clean(CSV_PATH)
    meter = build_dim_meter(100).iloc[0]

    rng = np.random.default_rng(42)
    noise = rng.normal(1.0, 0.05, size=len(clean_df))
    meter_load = np.clip(clean_df["load_kw"].values * meter["load_share"] * noise, 0.0, None)

    return pd.DataFrame({
        "meter_id": meter["meter_id"],
        "timestamp": clean_df["timestamp"].values,
        "date": clean_df["date"].values,
        "hour": clean_df["hour"].values,
        "load_kw": np.round(meter_load, 4),
    })


def reset_bench_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            DROP TABLE IF EXISTS {BENCH_TABLE};
            CREATE TABLE {BENCH_TABLE} (
                meter_id VARCHAR(16) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                date DATE NOT NULL,
                hour SMALLINT NOT NULL,
                load_kw DOUBLE PRECISION NOT NULL
            );
        """)
    conn.commit()


def bench_row_by_row_insert(conn, rows):
    sql = f"INSERT INTO {BENCH_TABLE} (meter_id, timestamp, date, hour, load_kw) VALUES (%s, %s, %s, %s, %s)"
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(sql, r)
    conn.commit()
    return time.perf_counter() - t0


def bench_batch_insert(conn, rows, page_size=1000):
    sql = f"INSERT INTO {BENCH_TABLE} (meter_id, timestamp, date, hour, load_kw) VALUES %s"
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=page_size)
    conn.commit()
    return time.perf_counter() - t0


def bench_copy(conn, df: pd.DataFrame):
    t0 = time.perf_counter()
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {BENCH_TABLE} FROM STDIN WITH (FORMAT CSV)", buf)
    conn.commit()
    return time.perf_counter() - t0


def main():
    sample_df = build_sample_batch()
    n_rows = len(sample_df)
    rows = list(sample_df.itertuples(index=False, name=None))

    conn = psycopg2.connect(**DB_CONFIG)

    methods = [
        ("Row-by-Row INSERT", lambda: bench_row_by_row_insert(conn, rows)),
        ("Batch INSERT (execute_values)", lambda: bench_batch_insert(conn, rows)),
        ("PostgreSQL COPY", lambda: bench_copy(conn, sample_df)),
    ]

    results = []
    for name, fn in methods:
        reset_bench_table(conn)
        elapsed = fn()
        rate = n_rows / elapsed
        est_full_sec = FULL_ROWS / rate
        results.append({
            "method": name,
            "sample_rows": n_rows,
            "elapsed_sec": round(elapsed, 4),
            "rows_per_sec": int(rate),
            "est_full_4.38M_sec": round(est_full_sec, 2),
        })

    # Clean up temp table
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {BENCH_TABLE};")
    conn.commit()
    conn.close()

    res_df = pd.DataFrame(results)
    baseline_time = res_df.loc[0, "elapsed_sec"]
    res_df["speedup_vs_insert"] = (baseline_time / res_df["elapsed_sec"]).round(1).astype(str) + "x"

    out_path = OUTPUT_DIR / "benchmark_load_results.csv"
    res_df.to_csv(out_path, index=False)

    print("\n--- INGESTION BENCHMARK (43,848 rows = 1 meter) ---")
    print(res_df.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()