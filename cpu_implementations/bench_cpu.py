#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECC Scalar Multiplication CPU Benchmark
Platform: Intel Core i9-14900K
Benchmarks sect163k1, sect233k1, sect571k1 scalar multiplication.
"""

import sys
import os
import io
import time
import datetime
import statistics

import psutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import the three scalar multiplication modules
# ---------------------------------------------------------------------------
import scalar_mult_163
import scalar_mult_233
import scalar_mult_571

# ---------------------------------------------------------------------------
# RAPL helpers (Intel power measurement)
# ---------------------------------------------------------------------------
RAPL_ENERGY_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"


def rapl_available():
    """Check if we can read the RAPL energy counter."""
    try:
        with open(RAPL_ENERGY_PATH, "r") as f:
            f.read()
        return True
    except (PermissionError, FileNotFoundError, OSError):
        return False


def read_rapl_uj():
    """Read current RAPL energy counter in micro-joules."""
    with open(RAPL_ENERGY_PATH, "r") as f:
        return int(f.read().strip())


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
WARMUP_RUNS = 2
TIMED_RUNS = 10
PLATFORM = "Intel Core i9-14900K"

# Define the three workloads: (label, module, function, k, G)
BENCHMARKS = [
    {
        "bits": 163,
        "label": "sect163k1",
        "script": "scalar_mult_163.py",
        "func": scalar_mult_163.ecc_gf2m_163,
        "k": 0x7FFF705303B81F7440ECD05E56F02D89DB3D00E14,
        "G": (scalar_mult_163.Gx, scalar_mult_163.Gy, 1),
        "out_file": "cpu_i9_163_stats.txt",
    },
    {
        "bits": 233,
        "label": "sect233k1",
        "script": "scalar_mult_233.py",
        "func": scalar_mult_233.ecc_gf2m_233,
        "k": 0x1673B01DC2D7D593D7DD6C2A9E2D8C427898A000F91B6F924A467ED232A,
        "G": (scalar_mult_233.Gx, scalar_mult_233.Gy, 1),
        "out_file": "cpu_i9_233_stats.txt",
    },
    {
        "bits": 571,
        "label": "sect571k1",
        "script": "scalar_mult_571.py",
        "func": scalar_mult_571.ecc_gf2m_571,
        "k": 0x7FFF0BB344B6439B0803C471F95F0DA54DCAFD6FD920DE4BC70CE0B2A4B2ACF3DADD796AA4B468C9F3AD9F525489758D7617D584A33F145C1E5FF8AAC1BBF5B082CD6253449B27F,
        "G": (scalar_mult_571.Gx, scalar_mult_571.Gy, 1),
        "out_file": "cpu_i9_571_stats.txt",
    },
]


def run_scalar_mult_silent(func, k, G):
    """Run the scalar multiplication while suppressing all stdout prints."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = func(k, G)
    finally:
        sys.stdout = old_stdout
    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    use_rapl = rapl_available()
    if use_rapl:
        print("[INFO] RAPL power measurement available.")
    else:
        print("[INFO] RAPL not available — power will be reported as N/A.")

    # Initialize psutil CPU measurement (first call always returns 0)
    psutil.cpu_percent(interval=None)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_latencies = {}  # bits -> list of latencies

    for bench in BENCHMARKS:
        bits = bench["bits"]
        label = bench["label"]
        script = bench["script"]
        func = bench["func"]
        k = bench["k"]
        G = bench["G"]
        out_file = bench["out_file"]

        print(f"\n{'='*50}")
        print(f"  Benchmarking {script}  ({label})")
        print(f"{'='*50}")

        # ---- Warm-up ----
        for w in range(1, WARMUP_RUNS + 1):
            print(f"  Warm-up {w}/{WARMUP_RUNS} ...", end=" ", flush=True)
            run_scalar_mult_silent(func, k, G)
            print("done")

        # ---- Timed runs ----
        runs = []  # list of dicts per run

        for r in range(1, TIMED_RUNS + 1):
            print(f"  Running {script} — Run {r}/{TIMED_RUNS} ...", end=" ", flush=True)

            # Read RAPL before
            rapl_before = read_rapl_uj() if use_rapl else None

            t_start = time.perf_counter()
            run_scalar_mult_silent(func, k, G)
            t_end = time.perf_counter()

            # Read RAPL after
            rapl_after = read_rapl_uj() if use_rapl else None

            latency_s = t_end - t_start
            latency_ms = latency_s * 1000.0

            cpu_util = psutil.cpu_percent(interval=None)
            ram_mb = psutil.virtual_memory().used / (1024 ** 2)

            if use_rapl and rapl_before is not None and rapl_after is not None:
                energy_uj = rapl_after - rapl_before
                if energy_uj < 0:
                    # Counter wrapped around
                    energy_uj += 2**32
                power_w = (energy_uj / 1e6) / latency_s if latency_s > 0 else 0.0
                energy_mj = power_w * latency_s * 1000.0
            else:
                power_w = None
                energy_mj = None

            runs.append({
                "latency_ms": latency_ms,
                "cpu_util": cpu_util,
                "ram_mb": ram_mb,
                "power_w": power_w,
                "energy_mj": energy_mj,
            })

            print(f"done  ({latency_ms:.2f} ms)")

        latencies = [r["latency_ms"] for r in runs]
        all_latencies[bits] = latencies

        # ---- Compute summary stats ----
        mean_lat = statistics.mean(latencies)
        std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        min_lat = min(latencies)
        max_lat = max(latencies)
        med_lat = statistics.median(latencies)
        avg_cpu = statistics.mean([r["cpu_util"] for r in runs])
        avg_ram = statistics.mean([r["ram_mb"] for r in runs])

        if runs[0]["power_w"] is not None:
            avg_power = statistics.mean([r["power_w"] for r in runs])
            avg_energy = statistics.mean([r["energy_mj"] for r in runs])
            power_str = f"{avg_power:.1f} W"
            energy_str = f"{avg_energy:.3f} mJ"
        else:
            avg_power = None
            avg_energy = None
            power_str = "N/A"
            energy_str = "N/A"

        # ---- Write stats file ----
        lines = []
        lines.append("============================================")
        lines.append("  ECC CPU Benchmark Results")
        lines.append("============================================")
        lines.append(f"Script:       {script}")
        lines.append(f"Platform:     {PLATFORM}")
        lines.append(f"Date:         {now_str}")
        lines.append(f"Warm-up runs: {WARMUP_RUNS}")
        lines.append(f"Timed runs:   {TIMED_RUNS}")
        lines.append("--------------------------------------------")
        lines.append(f"{'Run':>3}  {'Latency(ms)':>11}  {'CPU_Util(%)':>11}  {'RAM(MB)':>8}  {'Power(W)':>8}  {'Energy(mJ)':>10}")

        for i, r in enumerate(runs, 1):
            pw = f"{r['power_w']:.1f}" if r["power_w"] is not None else "N/A"
            en = f"{r['energy_mj']:.3f}" if r["energy_mj"] is not None else "N/A"
            lines.append(
                f"{i:>3}  {r['latency_ms']:>11.2f}  {r['cpu_util']:>11.1f}  "
                f"{r['ram_mb']:>8.1f}  {pw:>8}  {en:>10}"
            )

        lines.append("--------------------------------------------")
        lines.append("Summary Statistics:")
        lines.append(f"  Mean Latency:   {mean_lat:.2f} ms")
        lines.append(f"  Std Dev:        {std_lat:.2f} ms")
        lines.append(f"  Min:            {min_lat:.2f} ms")
        lines.append(f"  Max:            {max_lat:.2f} ms")
        lines.append(f"  Median:         {med_lat:.2f} ms")
        lines.append(f"  Avg CPU Util:   {avg_cpu:.1f} %")
        lines.append(f"  Avg RAM:        {avg_ram:.1f} MB")
        lines.append(f"  Avg Power:      {power_str}")
        lines.append(f"  Avg Energy:     {energy_str}")
        lines.append("============================================")

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_file)
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"  -> Saved {out_file}")

    # ------------------------------------------------------------------
    # Generate latency plot
    # ------------------------------------------------------------------
    print("\nGenerating latency plot ...")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(1, TIMED_RUNS + 1))

    colors = {163: "blue", 233: "orange", 571: "red"}
    labels = {163: "sect163k1", 233: "sect233k1", 571: "sect571k1"}

    for bits in [163, 233, 571]:
        lat = all_latencies[bits]
        mean_val = statistics.mean(lat)
        ax.plot(x, lat, marker="o", color=colors[bits], label=labels[bits])
        ax.axhline(y=mean_val, color=colors[bits], linestyle="--", alpha=0.5,
                    label=f"{labels[bits]} mean ({mean_val:.1f} ms)")

    ax.set_xlabel("Run Number")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("ECC Scalar Multiplication Latency \u2014 Intel i9-14900K")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "cpu_i9_latency_plot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"  -> Saved cpu_i9_latency_plot.png")
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
