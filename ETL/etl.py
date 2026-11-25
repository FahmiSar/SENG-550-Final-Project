import os
import pandas as pd
import numpy as np
from pathlib import Path

ETL_DIR = Path(__file__).resolve().parent
CSV_PATH = ETL_DIR.parent / "ML" / "smartgrid_5yr_hourly_natural-selected.csv"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_and_clean(path):
    df = pd.read_csv(path)
    raw_rows = len(df)

    # Quality checks
    nulls = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())
    neg_loads = int((df["load_kw"] < 0).sum())

    # Clean & Transform
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    # Check for missing hourly gaps
    expected_hours = int((df["timestamp"].max() - df["timestamp"].min()).total_seconds() // 3600) + 1
    missing_hours = expected_hours - len(df)

    # Drop zero-variance column
    if "is_lockdown" in df.columns:
        df = df.drop(columns=["is_lockdown"])

    # Combine 3 season booleans into 1 text column
    df["season"] = "harmattan"
    df.loc[df["season_dry"], "season"] = "dry"
    df.loc[df["season_rainy"], "season"] = "rainy"
    df = df.drop(columns=["season_dry", "season_rainy", "season_harmattan"])

    # Save quality report
    report = pd.DataFrame([
        {"metric": "raw_rows", "value": raw_rows},
        {"metric": "clean_rows", "value": len(df)},
        {"metric": "null_values", "value": nulls},
        {"metric": "duplicate_rows", "value": dups},
        {"metric": "negative_load_values", "value": neg_loads},
        {"metric": "missing_hourly_timestamps", "value": missing_hours},
        {"metric": "dropped_columns", "value": "is_lockdown"},
    ])
    report_path = os.path.join(OUTPUT_DIR, "data_quality_report.csv")
    report.to_csv(report_path, index=False)

    print(report.to_string(index=False))
    print(f"\nSaved to {report_path}")
    return df

def build_meters(num_meters=100, seed=42):
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
        "load_share": shares,
    })

if __name__ == "__main__":
    clean_df = extract_and_clean(CSV_PATH)
    meters_df = build_meters(100)
    print("\n--- SMART METERS ---")
    print(meters_df["consumer_type"].value_counts())
    print(meters_df.head())
    print(f"\nTotal rows when expanded: {len(clean_df) * len(meters_df):,}")
