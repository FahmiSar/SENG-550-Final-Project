import os
import numpy as np
import pandas as pd

from data_loader import data_loader_postgres
from models import SmartGridForecaster


def detect_columns(df):
    """Auto-detects the timestamp column and primary load target column."""
    cols = df.columns.tolist()

    time_candidates = [
        c for c in cols if any(k in c for k in ("time", "date", "stamp"))
    ]
    if not time_candidates:
        raise KeyError(f"No timestamp column found in {cols}")
    time_col = time_candidates[0]

    load_candidates = [
        c for c in cols if any(k in c for k in ("load", "consumption", "demand", "kw", "mw"))
    ]
    if not load_candidates:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        load_col = numeric_cols[-1]
    else:
        total_cols = [c for c in load_candidates if "total" in c]
        load_col = total_cols[0] if total_cols else load_candidates[0]

    return time_col, load_col


def engineer_features(df, time_col, target_col):
    """
    Builds leakage-free cyclical calendar features, autoregressive load lags,
    rolling statistics, and thermal degree-day proxies.
    """
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)

    hour = data[time_col].dt.hour
    dow = data[time_col].dt.dayofweek
    month = data[time_col].dt.month

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    data["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    data["is_weekend"] = (dow >= 5).astype(int)

    data["load_lag_1"] = data[target_col].shift(1)
    data["load_lag_24"] = data[target_col].shift(24)
    data["load_lag_168"] = data[target_col].shift(168)

    shifted_target = data[target_col].shift(1)
    data["load_roll_mean_24"] = shifted_target.rolling(window=24).mean()
    data["load_roll_std_24"] = shifted_target.rolling(window=24).std()

    temp_cols = [c for c in data.columns if "temp" in c]
    if temp_cols:
        t_col = temp_cols[0]
        data["heating_degree_hours"] = np.maximum(0.0, 18.0 - data[t_col])
        data["cooling_degree_hours"] = np.maximum(0.0, data[t_col] - 22.0)

    data = data.dropna().reset_index(drop=True)

    exclude_cols = {time_col, target_col}
    other_load_cols = [
        c
        for c in data.columns
        if c not in exclude_cols
        and not c.startswith("load_")
        and any(k in c for k in ("load", "consumption", "demand", "kw", "mw"))
    ]
    exclude_cols.update(other_load_cols)

    feature_cols = [
        c
        for c in data.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    X = data[feature_cols]
    y = data[target_col]
    return X, y


def main():

    df = data_loader_postgres()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    time_col, target_col = detect_columns(df)
    print(f"[INFO] Detected timestamp column: '{time_col}' | Target: '{target_col}'")

    X, y = engineer_features(df, time_col=time_col, target_col=target_col)
    print(f"[INFO] Engineered {X.shape[1]} features across {X.shape[0]} rows:")
    print(f"       {X.columns.tolist()}\n")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    forecaster = SmartGridForecaster(random_state=42)
    print("=== MODEL ACCURACY & LATENCY BENCHMARK ===")
    results_df = forecaster.train_and_evaluate_all(X_train, y_train, X_test, y_test)
    print(results_df.to_string(index=False))

    print("\n=== MULTI-METER SCALABILITY BENCHMARK (HistGradientBoosting) ===")
    scale_df = forecaster.benchmark_scalability(
    X, y, model_name="Hist_Gradient_Boosting"
    )
    print(scale_df.to_string(index=False))

    os.makedirs("ML/outputs", exist_ok=True)
    results_df.to_csv("ML/outputs/model_comparison.csv", index=False)
    scale_df.to_csv("ML/outputs/scalability_benchmark.csv", index=False)
    print("\n[DONE] Saved benchmark CSVs to ML/outputs/")


if __name__ == "__main__":
    main()