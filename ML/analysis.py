
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_CSV = BASE_DIR / "outputs" / "model_comparison.csv"
SCALABILITY_CSV = BASE_DIR / "outputs" / "scalability_benchmark.csv"

OUTPUT_DIR = BASE_DIR / "benchmark_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "benchmark_summary.txt"

# --------------------------------------------------
# Load data
# --------------------------------------------------

models = pd.read_csv(MODEL_CSV)
scalability = pd.read_csv(SCALABILITY_CSV)

models["model_label"] = models["model"].str.replace("_", " ")

# --------------------------------------------------
# Helper function
# --------------------------------------------------

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# --------------------------------------------------
# 1. Model Accuracy Comparison
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(models["model_label"], models["MAE"])
axes[0].set_title("Mean Absolute Error (MAE)")
axes[0].set_ylabel("MAE (kW)")
axes[0].tick_params(axis="x", rotation=35)

axes[1].bar(models["model_label"], models["RMSE"])
axes[1].set_title("Root Mean Squared Error (RMSE)")
axes[1].set_ylabel("RMSE (kW)")
axes[1].tick_params(axis="x", rotation=35)

fig.suptitle("Forecasting Accuracy Comparison", fontsize=15)

save_plot("01_model_accuracy.png")


# --------------------------------------------------
# 2. Model Training and Inference Time
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(models))
width = 0.35

ax.bar(
    [i - width / 2 for i in x],
    models["train_time_sec"],
    width,
    label="Training Time"
)

ax.bar(
    [i + width / 2 for i in x],
    models["infer_time_sec"],
    width,
    label="Inference Time"
)

ax.set_xticks(list(x))
ax.set_xticklabels(models["model_label"], rotation=30, ha="right")
ax.set_ylabel("Time (seconds)")
ax.set_title("Model Training vs Inference Time")
ax.legend()

save_plot("02_training_inference_time.png")


# --------------------------------------------------
# 3. Model Memory Consumption
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    models["model_label"],
    models["peak_train_mem_mb"]
)

ax.set_title("Peak Training Memory Consumption")
ax.set_ylabel("Memory (MB)")
ax.tick_params(axis="x", rotation=30)

save_plot("03_model_memory.png")


# --------------------------------------------------
# 4. Model Inference Throughput
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    models["model_label"],
    models["infer_rows_per_sec"]
)

ax.set_title("Model Inference Throughput")
ax.set_ylabel("Rows per second")
ax.tick_params(axis="x", rotation=30)

save_plot("04_inference_throughput.png")


# --------------------------------------------------
# 5. Scalability: Training Time and Memory
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    scalability["total_rows"],
    scalability["train_time_sec"],
    marker="o"
)

axes[0].set_title("Training Time vs Dataset Size")
axes[0].set_xlabel("Number of Rows")
axes[0].set_ylabel("Training Time (seconds)")
axes[0].grid(True)

axes[1].plot(
    scalability["total_rows"],
    scalability["peak_mem_mb"],
    marker="o"
)

axes[1].set_title("Memory Usage vs Dataset Size")
axes[1].set_xlabel("Number of Rows")
axes[1].set_ylabel("Peak Memory (MB)")
axes[1].grid(True)

fig.suptitle("Hist Gradient Boosting Scalability", fontsize=15)

save_plot("05_scalability_time_memory.png")


# --------------------------------------------------
# 6. Scalability: Training Throughput
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    scalability["data_fraction"] * 100,
    scalability["train_throughput_rows_sec"],
    marker="o"
)

ax.set_title("Training Throughput vs Dataset Size")
ax.set_xlabel("Dataset Size (%)")
ax.set_ylabel("Training Throughput (rows/sec)")
ax.grid(True)

save_plot("06_scalability_throughput.png")


# --------------------------------------------------
# 7. Generate Benchmark Summary
# --------------------------------------------------

best_mae = models.loc[models["MAE"].idxmin()]
fastest_training = models.loc[models["train_time_sec"].idxmin()]
fastest_inference = models.loc[models["infer_rows_per_sec"].idxmax()]
lowest_memory = models.loc[models["peak_train_mem_mb"].idxmin()]

summary = []

summary.append("MODEL COMPARISON SUMMARY")
summary.append("=" * 40)

summary.append(f"Best MAE: {best_mae['model']} ({best_mae['MAE']:.4f})")
summary.append(f"Fastest Training: {fastest_training['model']} ({fastest_training['train_time_sec']:.4f} sec)")
summary.append(f"Fastest Inference: {fastest_inference['model']} ({fastest_inference['infer_rows_per_sec']:.2f} rows/sec)")
summary.append(f"Lowest Memory: {lowest_memory['model']} ({lowest_memory['peak_train_mem_mb']:.2f} MB)")

summary.append("")
summary.append("SCALABILITY SUMMARY")
summary.append("=" * 40)

summary.append(f"Largest Dataset: {scalability['total_rows'].max():,} rows")
summary.append(f"Final Training Time: {scalability.iloc[-1]['train_time_sec']:.4f} sec")
summary.append(f"Final Peak Memory: {scalability.iloc[-1]['peak_mem_mb']:.2f} MB")
summary.append(
    f"Final Training Throughput: "
    f"{scalability.iloc[-1]['train_throughput_rows_sec']:.2f} rows/sec"
)


# Write summary to text file
summary_text = "\n".join(summary)

SUMMARY_FILE.write_text(summary_text, encoding="utf-8")

print("Benchmark analysis completed.")
print(f"Charts saved to: {OUTPUT_DIR}")
print(f"Summary saved to: {SUMMARY_FILE}")
