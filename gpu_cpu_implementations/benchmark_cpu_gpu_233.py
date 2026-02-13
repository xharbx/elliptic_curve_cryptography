# -*- coding: utf-8 -*-
"""
Benchmark CPU vs GPU scalar multiplication for GF(2^233).
Runs each 10 times, plots results, and saves statistics.
"""

import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import os

DIR = os.path.dirname(os.path.abspath(__file__))
CPU_SCRIPT = os.path.join(DIR, "scalar_mult_233.py")
GPU_SCRIPT = os.path.join(DIR, "GPU_kernal_ecc_scalar_233.py")
NUM_RUNS = 10

# --- Run CPU benchmark ---
cpu_times = []
print("=" * 50)
print("Running CPU benchmark (GF(2^233))")
print("=" * 50)
for run in range(1, NUM_RUNS + 1):
    result = subprocess.run(["python3", CPU_SCRIPT], capture_output=True, text=True)
    match = re.search(r"Scalar multiplication completed in\s+([\d.]+)\s+ms", result.stdout)
    t = float(match.group(1)) if match else 0
    cpu_times.append(t)
    print(f"  CPU Run {run:2d}: {t:.2f} ms")

# --- Run GPU benchmark ---
gpu_run_avgs = []
print("\n" + "=" * 50)
print("Running GPU benchmark (GF(2^233))")
print("=" * 50)
for run in range(1, NUM_RUNS + 1):
    result = subprocess.run(["python3", GPU_SCRIPT], capture_output=True, text=True)
    times = [float(m) for m in re.findall(r"GPU runtime\s*=\s*([\d.]+)\s*ms", result.stdout)]
    avg = np.mean(times) if times else 0
    gpu_run_avgs.append(avg)
    print(f"  GPU Run {run:2d}: {avg:.2f} ms (avg of {len(times)} tests)")

# --- Statistics ---
lines = []
lines.append("=" * 65)
lines.append("CPU vs GPU Benchmark - GF(2^233) Scalar Multiplication")
lines.append("=" * 65)
lines.append(f"Number of runs: {NUM_RUNS}")
lines.append("")

lines.append("-" * 65)
lines.append("Per-Run Times (ms)")
lines.append("-" * 65)
lines.append(f"{'Run':<6} {'CPU (ms)':>12} {'GPU (ms)':>12} {'Speedup':>10}")
lines.append("-" * 65)
for i in range(NUM_RUNS):
    speedup = cpu_times[i] / gpu_run_avgs[i] if gpu_run_avgs[i] else 0
    lines.append(f"{i+1:<6} {cpu_times[i]:>12.2f} {gpu_run_avgs[i]:>12.2f} {speedup:>9.2f}x")

lines.append("")
lines.append("-" * 65)
lines.append("CPU Statistics")
lines.append("-" * 65)
lines.append(f"  Mean:          {np.mean(cpu_times):.2f} ms")
lines.append(f"  Median:        {np.median(cpu_times):.2f} ms")
lines.append(f"  Std Deviation: {np.std(cpu_times):.2f} ms")
lines.append(f"  Min:           {np.min(cpu_times):.2f} ms")
lines.append(f"  Max:           {np.max(cpu_times):.2f} ms")

lines.append("")
lines.append("-" * 65)
lines.append("GPU Statistics")
lines.append("-" * 65)
lines.append(f"  Mean:          {np.mean(gpu_run_avgs):.2f} ms")
lines.append(f"  Median:        {np.median(gpu_run_avgs):.2f} ms")
lines.append(f"  Std Deviation: {np.std(gpu_run_avgs):.2f} ms")
lines.append(f"  Min:           {np.min(gpu_run_avgs):.2f} ms")
lines.append(f"  Max:           {np.max(gpu_run_avgs):.2f} ms")

lines.append("")
lines.append("-" * 65)
lines.append("Speedup (CPU / GPU)")
lines.append("-" * 65)
avg_speedup = np.mean(cpu_times) / np.mean(gpu_run_avgs) if np.mean(gpu_run_avgs) else 0
lines.append(f"  Average Speedup: {avg_speedup:.2f}x")
lines.append("=" * 65)

stats_text = "\n".join(lines)
print("\n" + stats_text)

stats_path = os.path.join(DIR, "benchmark_cpu_gpu_233_statistics.txt")
with open(stats_path, "w") as f:
    f.write(stats_text + "\n")
print(f"\nStatistics saved to: {stats_path}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
runs = list(range(1, NUM_RUNS + 1))

# Left: CPU vs GPU line chart
axes[0].plot(runs, cpu_times, "o-", color="#E53935", linewidth=2, markersize=8, label=f"CPU (Mean: {np.mean(cpu_times):.2f} ms)")
axes[0].plot(runs, gpu_run_avgs, "s-", color="#2196F3", linewidth=2, markersize=8, label=f"GPU (Mean: {np.mean(gpu_run_avgs):.2f} ms)")
axes[0].set_xlabel("Run", fontsize=12)
axes[0].set_ylabel("Time (ms)", fontsize=12)
axes[0].set_title("GF(2^233) Scalar Multiplication\nCPU vs GPU Time per Run", fontsize=13)
axes[0].set_xticks(runs)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Right: Bar chart comparison
x = np.arange(NUM_RUNS)
width = 0.35
axes[1].bar(x - width/2, cpu_times, width, label="CPU", color="#E53935", alpha=0.85)
axes[1].bar(x + width/2, gpu_run_avgs, width, label="GPU", color="#2196F3", alpha=0.85)
axes[1].set_xlabel("Run", fontsize=12)
axes[1].set_ylabel("Time (ms)", fontsize=12)
axes[1].set_title("GF(2^233) Scalar Multiplication\nCPU vs GPU Comparison", fontsize=13)
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(i) for i in runs])
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = os.path.join(DIR, "benchmark_cpu_gpu_233_plot.png")
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to: {plot_path}")
plt.close()
