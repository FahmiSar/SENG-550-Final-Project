import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class NaivePersistenceModel:
  """Baseline 1: Predicts yesterday's load at the exact same hour (t - 24)."""

  def __init__(self, lag_col: str = "load_lag_24"):
    self.lag_col = lag_col

  def fit(self, X: pd.DataFrame, y: pd.Series):
    if self.lag_col not in X.columns:
      raise KeyError(
          f"Expected '{self.lag_col}' in features for NaivePersistenceModel."
      )
    return self

  def predict(self, X: pd.DataFrame) -> np.ndarray:
    return X[self.lag_col].to_numpy()


class SmartGridForecaster:
  """
    Model registry and benchmarking wrapper for smart grid load forecasting.
    Supports baselines, linear models, gradient boosted trees, and neural nets.
    """

  def __init__(self, random_state: int = 42):
    self.random_state = random_state
    self.models = self._build_model_suite()
    self.fitted_models = {}

  def _build_model_suite(self) -> dict:
    return {
        "Naive_Persistence_24h": NaivePersistenceModel(lag_col="load_lag_24"),
        "Ridge_Regression": make_pipeline(
            StandardScaler(), Ridge(alpha=1.0, random_state=self.random_state)
        ),
        "Random_Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            n_jobs=-1,
            random_state=self.random_state,
        ),
        "Hist_Gradient_Boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=63,
            random_state=self.random_state,
        ),
        "MLP_Neural_Net": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=50,
                early_stopping=True,
                random_state=self.random_state,
            ),
        ),
    }

  def get_model_names(self) -> list[str]:
    return list(self.models.keys())

  def train_and_evaluate_all(
      self,
      X_train: pd.DataFrame,
      y_train: pd.Series,
      X_test: pd.DataFrame,
      y_test: pd.Series,
  ) -> pd.DataFrame:
    """Trains every model in the suite and returns accuracy + compute metrics."""
    results = []

    for name, model in self.models.items():
      # Measure peak memory and wall-clock training time
      tracemalloc.start()
      t0 = time.perf_counter()
      model.fit(X_train, y_train)
      train_time = time.perf_counter() - t0
      _, peak_mem_bytes = tracemalloc.get_traced_memory()
      tracemalloc.stop()

      self.fitted_models[name] = model

      # Measure inference latency & throughput
      t1 = time.perf_counter()
      preds = model.predict(X_test)
      infer_time = time.perf_counter() - t1

      mae = mean_absolute_error(y_test, preds)
      rmse = root_mean_squared_error(y_test, preds)
      mape = mean_absolute_percentage_error(y_test, preds) * 100.0
      throughput = len(X_test) / max(infer_time, 1e-9)

      results.append({
          "model": name,
          "MAE": round(mae, 4),
          "RMSE": round(rmse, 4),
          "MAPE_pct": round(mape, 4),
          "train_time_sec": round(train_time, 4),
          "infer_time_sec": round(infer_time, 4),
          "infer_rows_per_sec": round(throughput, 1),
          "peak_train_mem_mb": round(peak_mem_bytes / (1024 * 1024), 2),
      })

    return (
        pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    )

  def benchmark_scalability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "Hist_Gradient_Boosting",
        fractions: list[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
    ) -> pd.DataFrame:
    """
        Benchmarks training time, inference throughput, and peak RAM across
        increasing observation windows (e.g., 1 yr -> 5 yrs of historical data).
        """
    scaling_logs = []

    for frac in fractions:
        n_rows = int(len(X) * frac)
        X_sub, y_sub = X.iloc[:n_rows], y.iloc[:n_rows]

        split_idx = int(n_rows * 0.8)
        X_tr, X_te = X_sub.iloc[:split_idx], X_sub.iloc[split_idx:]
        y_tr, y_te = y_sub.iloc[:split_idx], y_sub.iloc[split_idx:]

        model = self._build_model_suite()[model_name]

        tracemalloc.start()
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_sec = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        t1 = time.perf_counter()
        _ = model.predict(X_te)
        infer_sec = time.perf_counter() - t1

        scaling_logs.append({
            "model": model_name,
            "data_fraction": frac,
            "total_rows": n_rows,
            "train_time_sec": round(train_sec, 4),
            "infer_time_sec": round(infer_sec, 4),
            "train_throughput_rows_sec": round(
                len(X_tr) / max(train_sec, 1e-9), 1
            ),
            "peak_mem_mb": round(peak_bytes / (1024 * 1024), 2),
        })

    return pd.DataFrame(scaling_logs)