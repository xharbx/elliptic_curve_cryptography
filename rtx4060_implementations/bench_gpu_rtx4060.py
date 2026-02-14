#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECC GPU Benchmark — NVIDIA RTX 4060
Benchmarks CuPy CUDA scalar multiplication kernels for:
  - sect163k1 (GF(2^163))
  - sect233k1 (GF(2^233))
  - sect571k1 (GF(2^571))

Usage: python bench_gpu_rtx4060.py
Author: Salah Harb
"""

import cupy as cp
import numpy as np
import time
import random
import subprocess
import os
import sys
import statistics
import datetime
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================
WARMUP_RUNS = 2
TIMED_RUNS = 10
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Curve Configurations
# ============================================================
CURVES = {
    '163': {
        'script': 'GPU_kernal_ecc_scalar_163.py',
        'kernel_name': 'scalar_mult_163',
        'M': 163,
        'n_limbs': 3,
        'a': 1,
        'b': 0x20A601907B8C953CA1481EB10512F78744A3205FD,
        'Gx': 0x03f0eba16286a2d57ea0991168d4994637e8343e36,
        'Gy': 0x0d51fbc6c71a0094fa2cdd545b11c5c0c797324f1,
        'Gz': 1,
        'label': 'sect163k1',
        'color': 'blue',
    },
    '233': {
        'script': 'GPU_kernal_ecc_scalar_233.py',
        'kernel_name': 'scalar_mult_233',
        'M': 233,
        'n_limbs': 4,
        'a': 1,
        'b': 0x066647ede6c332c7f8c0923bb58213b333b20e9ce4281fe115f7d8f90ad,
        'Gx': 0x0fac9dfcbac8313bb2139f1bb755fef65bc391f8b36f8f8eb7371fd558b,
        'Gy': 0x1006a08a41903350678e58528bebf8a0beff867a7ca36716f7e01f81052,
        'Gz': 1,
        'label': 'sect233k1',
        'color': 'orange',
    },
    '571': {
        'script': 'GPU_kernal_ecc_scalar_571.py',
        'kernel_name': 'scalar_mult_571',
        'M': 571,
        'n_limbs': 9,
        'a': 1,
        'b': 0x02f40e7e2221f295de297117b7f3d62f5c6a97ffcb8ceff1cd6ba8ce4a9a18ad84ffabbd8efa59332be7ad6756a66e294afd185a78ff12aa520e4de739baca0c7ffeff7f2955727a,
        'Gx': 0x303001d34b856296c16c0d40d3cd7750a93d1d2955fa80aa5f40fc8db7b2abdbde53950f4c0d293cdd711a35b67fb1499ae60038614f1394abfa3b4c850d927e1e7769c8eec2d19,
        'Gy': 0x37bf27342da639b6dccfffeb73d69d78c6c27a6009cbbca1980f8533921e8a684423e43bab08a576291af8f461bb2a8b3531d2f0485c19b16e2f1516e23dd3c1a4827af1b8ac15b,
        'Gz': 1,
        'label': 'sect571k1',
        'color': 'red',
    },
}


# ============================================================
# Helper: nvidia-smi GPU stats
# ============================================================
def get_gpu_stats():
    """Query nvidia-smi for GPU stats. Returns dict with float values or None on failure."""
    stats = {}
    queries = {
        'gpu_util': 'utilization.gpu',
        'mem_used': 'memory.used',
        'temperature': 'temperature.gpu',
        'power_draw': 'power.draw',
        'clock_sm': 'clocks.current.sm',
    }
    for key, query in queries.items():
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            stats[key] = float(result.stdout.strip())
        except Exception:
            stats[key] = None
    return stats


# ============================================================
# Helper: Extract kernel_code from script file
# ============================================================
def extract_kernel_code(filepath):
    """Extract the CUDA kernel_code raw string from a GPU script file."""
    with open(filepath, 'r') as f:
        content = f.read()
    marker = "kernel_code = r'''"
    start_idx = content.index(marker) + len(marker)
    end_idx = content.index("'''", start_idx)
    return content[start_idx:end_idx]


# ============================================================
# Helper: Integer to n-limb uint64 array
# ============================================================
def to_limbs(x, n_limbs):
    """Convert a Python integer to an n-element numpy uint64 array (little-endian limbs)."""
    return np.array([
        np.uint64((x >> (64 * i)) & ((1 << 64) - 1)) for i in range(n_limbs)
    ], dtype=np.uint64)


# ============================================================
# Helper: Suppress C-level stdout (CUDA kernel printf output)
# ============================================================
class SuppressStdout:
    """Context manager to suppress C-level stdout so CUDA kernel printf
    does not flood the console during benchmarking."""
    def __enter__(self):
        sys.stdout.flush()
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._old_stdout = os.dup(1)
        os.dup2(self._devnull, 1)
        return self

    def __exit__(self, *args):
        os.dup2(self._old_stdout, 1)
        os.close(self._devnull)
        os.close(self._old_stdout)


# ============================================================
# Print GPU device info
# ============================================================
def print_gpu_info():
    """Print GPU device name, compute capability, and total memory."""
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"  Device:             {name}")
    print(f"  Compute Capability: {props['major']}.{props['minor']}")
    print(f"  Total Memory:       {props['totalGlobalMem'] / (1024 ** 2):.0f} MB")


# ============================================================
# Setup curve: compile kernel, prepare GPU arrays
# ============================================================
def setup_curve(cfg):
    """Compile CUDA kernel and prepare GPU arrays for a given curve."""
    script_path = os.path.join(SCRIPT_DIR, cfg['script'])
    kernel_code = extract_kernel_code(script_path)

    cuda_root = "/usr/local/cuda-12.6"
    include_paths = [f"{cuda_root}/include", f"{cuda_root}/include/crt"]
    options = ('--std=c++11',) + tuple(f'-I{p}' for p in include_paths)

    mod = cp.RawModule(code=kernel_code, options=options)
    kernel_fn = mod.get_function(cfg['kernel_name'])

    n = cfg['n_limbs']
    Px = cp.asarray(to_limbs(cfg['Gx'], n))
    Py = cp.asarray(to_limbs(cfg['Gy'], n))
    Pz = cp.asarray(to_limbs(cfg['Gz'], n))
    a_gpu = cp.asarray(to_limbs(cfg['a'], n))
    b_gpu = cp.asarray(to_limbs(cfg['b'], n))
    Xout = cp.zeros(n, dtype=cp.uint64)
    Yout = cp.zeros(n, dtype=cp.uint64)
    Zout = cp.zeros(n, dtype=cp.uint64)

    return kernel_fn, Px, Py, Pz, a_gpu, b_gpu, Xout, Yout, Zout


# ============================================================
# Run a single scalar multiplication with timing
# ============================================================
def run_kernel_once(kernel_fn, Px, Py, Pz, a_gpu, b_gpu, Xout, Yout, Zout, cfg):
    """Run one scalar multiplication with a random scalar.
    Returns (cuda_ms, wall_ms) — CUDA event time and wall-clock time.
    """
    M = cfg['M']
    n = cfg['n_limbs']

    # Generate random scalar with MSB set
    k_int = random.getrandbits(M - 1) | (1 << (M - 1))
    k_gpu = cp.asarray(to_limbs(k_int, n), dtype=cp.uint64)

    # Reset output arrays
    Xout[:] = 0
    Yout[:] = 0
    Zout[:] = 0

    # Create CUDA events for precise GPU kernel timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    # Synchronize GPU before timing
    cp.cuda.Device(0).synchronize()

    wall_start = time.perf_counter()

    with SuppressStdout():
        start_event.record()
        kernel_fn((1,), (1,), (Px, Py, Pz, a_gpu, b_gpu, k_gpu, Xout, Yout, Zout))
        end_event.record()
        end_event.synchronize()

    wall_end = time.perf_counter()

    cuda_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    wall_ms = (wall_end - wall_start) * 1000.0

    return cuda_ms, wall_ms


# ============================================================
# Benchmark a single curve
# ============================================================
def benchmark_curve(curve_key, cfg):
    """Run warm-up + timed benchmark for one curve. Returns list of run result dicts."""
    print(f"\n{'=' * 64}")
    print(f"  Benchmarking: {cfg['script']}  ({cfg['label']})")
    print(f"{'=' * 64}")

    # Compile kernel and prepare GPU arrays
    print(f"  Compiling CUDA kernel ({cfg['kernel_name']})...")
    kernel_fn, Px, Py, Pz, a_gpu, b_gpu, Xout, Yout, Zout = setup_curve(cfg)
    print(f"  Compilation done.")

    # Warm-up runs (not timed)
    print(f"  Warm-up ({WARMUP_RUNS} runs)...")
    for w in range(WARMUP_RUNS):
        run_kernel_once(kernel_fn, Px, Py, Pz, a_gpu, b_gpu, Xout, Yout, Zout, cfg)
        print(f"    Warm-up {w + 1}/{WARMUP_RUNS} done.")

    # Timed runs
    print(f"  Timed runs ({TIMED_RUNS} runs)...")
    results = []
    for i in range(TIMED_RUNS):
        cuda_ms, wall_ms = run_kernel_once(
            kernel_fn, Px, Py, Pz, a_gpu, b_gpu, Xout, Yout, Zout, cfg
        )

        # GPU stats via nvidia-smi (captured after kernel + sync)
        gpu_stats = get_gpu_stats()

        # Host stats
        cpu_util = psutil.cpu_percent(interval=None)
        ram_used = psutil.virtual_memory().used / (1024 ** 2)

        # Energy (mJ) = power_W × latency_s × 1000
        power_w = gpu_stats.get('power_draw')
        energy_mj = (power_w * (cuda_ms / 1000.0) * 1000.0) if power_w is not None else None

        run_data = {
            'run': i + 1,
            'latency_ms': cuda_ms,
            'wall_ms': wall_ms,
            'gpu_util': gpu_stats.get('gpu_util'),
            'mem_used': gpu_stats.get('mem_used'),
            'temperature': gpu_stats.get('temperature'),
            'power_draw': power_w,
            'clock_sm': gpu_stats.get('clock_sm'),
            'cpu_util': cpu_util,
            'ram_used': ram_used,
            'energy_mj': energy_mj,
        }
        results.append(run_data)

        print(f"  Benchmarking {cfg['script']} \u2014 Run {i + 1}/{TIMED_RUNS}... "
              f"{cuda_ms:.3f} ms")

    return results


# ============================================================
# Format helper for stats table
# ============================================================
def fmt(val, fmt_str):
    """Format a value; returns right-aligned 'N/A' if None."""
    if val is None:
        width = int(fmt_str.split('.')[0]) if '.' in fmt_str else int(fmt_str.rstrip('fds'))
        return f"{'N/A':>{width}}"
    return format(val, fmt_str)


# ============================================================
# Write stats file for one curve
# ============================================================
def write_stats_file(filename, cfg, results):
    """Write formatted benchmark results to a text file."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    latencies = [r['latency_ms'] for r in results]
    gpu_utils = [r['gpu_util'] for r in results if r['gpu_util'] is not None]
    mem_useds = [r['mem_used'] for r in results if r['mem_used'] is not None]
    temps     = [r['temperature'] for r in results if r['temperature'] is not None]
    powers    = [r['power_draw'] for r in results if r['power_draw'] is not None]
    clocks    = [r['clock_sm'] for r in results if r['clock_sm'] is not None]
    cpu_utils = [r['cpu_util'] for r in results]
    ram_useds = [r['ram_used'] for r in results]
    energies  = [r['energy_mj'] for r in results if r['energy_mj'] is not None]

    W = 100  # output line width
    lines = []
    lines.append("=" * W)
    lines.append("  ECC GPU Benchmark Results \u2014 NVIDIA RTX 4060")
    lines.append("=" * W)
    lines.append(f"Script:         {cfg['script']}")
    lines.append(f"Platform:       NVIDIA RTX 4060 (8GB GDDR6)")
    lines.append(f"Date:           {now}")
    lines.append(f"Warm-up runs:   {WARMUP_RUNS}")
    lines.append(f"Timed runs:     {TIMED_RUNS}")
    lines.append("-" * W)

    # Table header
    hdr = (f"{'Run':>3}  {'Latency(ms)':>11}  {'GPU_Util(%)':>11}  {'GPU_Mem(MB)':>11}  "
           f"{'Temp(\u00b0C)':>7}  {'Power(W)':>8}  {'Clock(MHz)':>10}  "
           f"{'CPU_Util(%)':>11}  {'RAM(MB)':>9}  {'Energy(mJ)':>10}")
    lines.append(hdr)

    for r in results:
        row = (f"{r['run']:>3}  "
               f"{fmt(r['latency_ms'],  '11.3f')}  "
               f"{fmt(r['gpu_util'],    '11.1f')}  "
               f"{fmt(r['mem_used'],    '11.1f')}  "
               f"{fmt(r['temperature'], '7.0f')}  "
               f"{fmt(r['power_draw'],  '8.1f')}  "
               f"{fmt(r['clock_sm'],    '10.0f')}  "
               f"{fmt(r['cpu_util'],    '11.1f')}  "
               f"{fmt(r['ram_used'],    '9.1f')}  "
               f"{fmt(r['energy_mj'],   '10.3f')}")
        lines.append(row)

    lines.append("-" * W)
    lines.append("Summary Statistics:")

    # Latency
    lines.append("  Latency:")
    lines.append(f"    Mean:       {statistics.mean(latencies):.3f} ms")
    lines.append(f"    Std Dev:    {statistics.stdev(latencies):.3f} ms"
                 if len(latencies) > 1 else "    Std Dev:    0.000 ms")
    lines.append(f"    Min:        {min(latencies):.3f} ms")
    lines.append(f"    Max:        {max(latencies):.3f} ms")
    lines.append(f"    Median:     {statistics.median(latencies):.3f} ms")

    # GPU Resources
    lines.append("  GPU Resources:")
    lines.append(f"    Avg GPU Util:     {statistics.mean(gpu_utils):.1f} %"
                 if gpu_utils else "    Avg GPU Util:     N/A")
    lines.append(f"    Avg GPU Memory:   {statistics.mean(mem_useds):.1f} MB"
                 if mem_useds else "    Avg GPU Memory:   N/A")
    lines.append(f"    Avg Temperature:  {statistics.mean(temps):.1f} \u00b0C"
                 if temps else "    Avg Temperature:  N/A")
    lines.append(f"    Avg Clock Speed:  {statistics.mean(clocks):.1f} MHz"
                 if clocks else "    Avg Clock Speed:  N/A")

    # Power & Energy
    lines.append("  Power & Energy:")
    lines.append(f"    Avg Power Draw:   {statistics.mean(powers):.1f} W"
                 if powers else "    Avg Power Draw:   N/A")
    lines.append(f"    Avg Energy/Run:   {statistics.mean(energies):.3f} mJ"
                 if energies else "    Avg Energy/Run:   N/A")
    lines.append(f"    Total Energy:     {sum(energies):.3f} mJ"
                 if energies else "    Total Energy:     N/A")

    # Host
    lines.append("  Host:")
    lines.append(f"    Avg CPU Util:     {statistics.mean(cpu_utils):.1f} %")
    lines.append(f"    Avg RAM Used:     {statistics.mean(ram_useds):.1f} MB")

    lines.append("=" * W)

    filepath = os.path.join(SCRIPT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Stats written to: {filename}")


# ============================================================
# Generate all plots
# ============================================================
def generate_plots(all_results):
    """Generate latency, power draw, and GPU utilization plots."""
    runs = list(range(1, TIMED_RUNS + 1))

    # ---- Plot 1: Latency ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        latencies = [r['latency_ms'] for r in all_results[key]]
        mean_lat = statistics.mean(latencies)
        ax.plot(runs, latencies, marker='o', color=cfg['color'], label=cfg['label'])
        ax.axhline(y=mean_lat, color=cfg['color'], linestyle='--', alpha=0.5,
                    label=f"{cfg['label']} mean ({mean_lat:.2f} ms)")
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('ECC Scalar Multiplication Latency \u2014 NVIDIA RTX 4060')
    ax.set_xticks(runs)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'rtx4060_latency_plot.png'), dpi=150)
    plt.close(fig)
    print(f"  Plot saved: rtx4060_latency_plot.png")

    # ---- Plot 2: Power Draw ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        powers = [r['power_draw'] if r['power_draw'] is not None else 0
                  for r in all_results[key]]
        ax.plot(runs, powers, marker='o', color=cfg['color'], label=cfg['label'])
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Power Draw (W)')
    ax.set_title('GPU Power Draw per Run \u2014 NVIDIA RTX 4060')
    ax.set_xticks(runs)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'rtx4060_power_plot.png'), dpi=150)
    plt.close(fig)
    print(f"  Plot saved: rtx4060_power_plot.png")

    # ---- Plot 3: GPU Utilization ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        utils = [r['gpu_util'] if r['gpu_util'] is not None else 0
                 for r in all_results[key]]
        ax.plot(runs, utils, marker='o', color=cfg['color'], label=cfg['label'])
    ax.set_xlabel('Run Number')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('GPU Utilization per Run \u2014 NVIDIA RTX 4060')
    ax.set_xticks(runs)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'rtx4060_gpu_util_plot.png'), dpi=150)
    plt.close(fig)
    print(f"  Plot saved: rtx4060_gpu_util_plot.png")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 64)
    print("  ECC GPU Benchmark \u2014 NVIDIA RTX 4060")
    print("=" * 64)
    print()

    # Print GPU device info for verification
    print("GPU Device Info:")
    print_gpu_info()
    print()

    # Initialize psutil CPU counter (first call always returns 0.0)
    psutil.cpu_percent(interval=None)

    all_results = {}
    stats_files = {
        '163': 'rtx4060_163_stats.txt',
        '233': 'rtx4060_233_stats.txt',
        '571': 'rtx4060_571_stats.txt',
    }

    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        results = benchmark_curve(key, cfg)
        all_results[key] = results
        write_stats_file(stats_files[key], cfg, results)

    # Generate all plots
    print(f"\n{'=' * 64}")
    print("  Generating plots...")
    print(f"{'=' * 64}")
    generate_plots(all_results)

    print(f"\n{'=' * 64}")
    print("  Benchmark complete!")
    print(f"{'=' * 64}")


if __name__ == '__main__':
    main()
