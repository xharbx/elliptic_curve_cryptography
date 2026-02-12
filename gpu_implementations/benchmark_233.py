# -*- coding: utf-8 -*-
"""
Benchmark GPU_kernal_ecc_scalar_233.py over 10 runs.
Plots runs vs times and saves statistics to a txt file.
"""

import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT = os.path.join(os.path.dirname(__file__), "GPU_kernal_ecc_scalar_233.py")
NUM_RUNS = 10

all_run_times = []
run_averages = []

for run in range(1, NUM_RUNS + 1):
    print(f"=== Run {run}/{NUM_RUNS} ===")
    result = subprocess.run(
        ["python3", SCRIPT],
        capture_output=True, text=True
    )
    output = result.stdout
    times = [float(m) for m in re.findall(r"GPU runtime\s*=\s*([\d.]+)\s*ms", output)]
    all_run_times.append(times)
    avg = np.mean(times) if times else 0
    run_averages.append(avg)
    print(f"  Tests in run: {len(times)}, Avg time: {avg:.2f} ms")

all_times = [t for run_times in all_run_times for t in run_times]

# --- Statistics ---
stats_lines = []
stats_lines.append("=" * 60)
stats_lines.append("GPU GF(2^233) Scalar Multiplication - Benchmark Statistics")
stats_lines.append("=" * 60)
stats_lines.append(f"Number of runs: {NUM_RUNS}")
stats_lines.append(f"Tests per run:  {len(all_run_times[0]) if all_run_times else 0}")
stats_lines.append(f"Total tests:    {len(all_times)}")
stats_lines.append("")
stats_lines.append("-" * 60)
stats_lines.append("Per-Run Average Times (ms)")
stats_lines.append("-" * 60)
for i, avg in enumerate(run_averages, 1):
    stats_lines.append(f"  Run {i:2d}: {avg:.2f} ms")
stats_lines.append("")
stats_lines.append("-" * 60)
stats_lines.append("Overall Statistics (across all individual test times)")
stats_lines.append("-" * 60)
stats_lines.append(f"  Mean:               {np.mean(all_times):.2f} ms")
stats_lines.append(f"  Median:             {np.median(all_times):.2f} ms")
stats_lines.append(f"  Std Deviation:      {np.std(all_times):.2f} ms")
stats_lines.append(f"  Variance:           {np.var(all_times):.4f} ms^2")
stats_lines.append(f"  Min:                {np.min(all_times):.2f} ms")
stats_lines.append(f"  Max:                {np.max(all_times):.2f} ms")
stats_lines.append(f"  Range:              {np.max(all_times) - np.min(all_times):.2f} ms")
q1 = np.percentile(all_times, 25)
q3 = np.percentile(all_times, 75)
stats_lines.append(f"  25th Percentile:    {q1:.2f} ms")
stats_lines.append(f"  75th Percentile:    {q3:.2f} ms")
stats_lines.append(f"  IQR:                {q3 - q1:.2f} ms")
stats_lines.append("")
stats_lines.append("-" * 60)
stats_lines.append("Per-Run Statistics (ms)")
stats_lines.append("-" * 60)
stats_lines.append(f"  Mean of run avgs:   {np.mean(run_averages):.2f} ms")
stats_lines.append(f"  Median of run avgs: {np.median(run_averages):.2f} ms")
stats_lines.append(f"  Std Dev of run avgs:{np.std(run_averages):.2f} ms")
stats_lines.append(f"  Min run avg:        {np.min(run_averages):.2f} ms")
stats_lines.append(f"  Max run avg:        {np.max(run_averages):.2f} ms")
stats_lines.append("=" * 60)

stats_text = "\n".join(stats_lines)
print("\n" + stats_text)

out_dir = os.path.dirname(__file__)
stats_path = os.path.join(out_dir, "benchmark_233_statistics.txt")
with open(stats_path, "w") as f:
    f.write(stats_text + "\n")
print(f"\nStatistics saved to: {stats_path}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

runs = list(range(1, NUM_RUNS + 1))
axes[0].plot(runs, run_averages, "o-", color="#2196F3", linewidth=2, markersize=8)
axes[0].axhline(np.mean(run_averages), color="red", linestyle="--", label=f"Mean: {np.mean(run_averages):.2f} ms")
axes[0].set_xlabel("Run", fontsize=12)
axes[0].set_ylabel("Average Time (ms)", fontsize=12)
axes[0].set_title("GF(2^233) GPU Scalar Multiplication\nAverage Time per Run", fontsize=13)
axes[0].set_xticks(runs)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(all_run_times, tick_labels=[str(i) for i in runs])
axes[1].set_xlabel("Run", fontsize=12)
axes[1].set_ylabel("Time (ms)", fontsize=12)
axes[1].set_title("GF(2^233) GPU Scalar Multiplication\nPer-Test Time Distribution by Run", fontsize=13)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(out_dir, "benchmark_233_plot.png")
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to: {plot_path}")
plt.close()
