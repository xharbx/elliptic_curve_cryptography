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
WARMUP_RUNS = 5
TIMED_RUNS = 50
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
        'hex_width': 41,
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
        'hex_width': 59,
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
        'hex_width': 143,
    },
}


# ============================================================
# Helper: nvidia-smi GPU stats (single query for efficiency)
# ============================================================
def get_gpu_stats():
    """Query nvidia-smi for GPU stats. Returns dict with float values or None on failure."""
    stats = {}
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.sm',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(',')]
        keys = ['gpu_util', 'mem_used', 'temperature', 'power_draw', 'clock_sm']
        for i, key in enumerate(keys):
            try:
                val = parts[i] if i < len(parts) else '[N/A]'
                stats[key] = float(val) if val != '[N/A]' else None
            except (ValueError, IndexError):
                stats[key] = None
    except Exception:
        for key in ['gpu_util', 'mem_used', 'temperature', 'power_draw', 'clock_sm']:
            stats[key] = None
    return stats


def get_idle_power():
    """Get GPU power draw at idle (via nvidia-smi). Returns W or None."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        val = result.stdout.strip()
        return float(val) if val != '[N/A]' else None
    except Exception:
        return None


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


def limbs_to_int(arr, n_limbs):
    """Convert n-element uint64 GPU array back to a Python integer."""
    val = 0
    for i in range(n_limbs):
        val |= int(arr[i].get()) << (64 * i)
    return val


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
# GPU memory usage tracking
# ============================================================
def get_gpu_memory():
    """Get CuPy memory pool usage."""
    mempool = cp.get_default_memory_pool()
    pinned_pool = cp.get_default_pinned_memory_pool()
    return {
        'used_bytes': mempool.used_bytes(),
        'total_bytes': mempool.total_bytes(),
        'pinned_blocks': pinned_pool.n_free_blocks(),
    }


# ============================================================
# Get detailed GPU device properties
# ============================================================
def get_device_props():
    """Get detailed GPU device properties."""
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    return {
        'name': name,
        'compute_major': props['major'],
        'compute_minor': props['minor'],
        'total_mem': props['totalGlobalMem'],
        'sm_count': props['multiProcessorCount'],
        'max_threads_per_sm': props['maxThreadsPerMultiProcessor'],
        'max_threads_per_block': props['maxThreadsPerBlock'],
        'max_blocks_per_sm': props.get('maxBlocksPerMultiProcessor', 'N/A'),
        'clock_rate_khz': props['clockRate'],
        'mem_clock_khz': props['memoryClockRate'],
        'l2_cache': props['l2CacheSize'],
        'warp_size': props.get('warpSize', 32),
        'shared_mem_per_block': props['sharedMemPerBlock'],
        'shared_mem_per_sm': props.get('sharedMemPerMultiprocessor', 0),
        'regs_per_block': props['regsPerBlock'],
        'regs_per_sm': props.get('regsPerMultiprocessor', 0),
    }


# ============================================================
# Print GPU device info
# ============================================================
def print_gpu_info(dev_props):
    """Print GPU device name, compute capability, and total memory."""
    print(f"  Device:             {dev_props['name']}")
    print(f"  Compute Capability: {dev_props['compute_major']}.{dev_props['compute_minor']}")
    print(f"  SMs:                {dev_props['sm_count']}")
    print(f"  GPU Clock:          {dev_props['clock_rate_khz']/1000:.0f} MHz")
    print(f"  Memory Clock:       {dev_props['mem_clock_khz']/1000:.0f} MHz")
    print(f"  Total Memory:       {dev_props['total_mem'] / (1024 ** 3):.2f} GB")
    print(f"  L2 Cache:           {dev_props['l2_cache'] / 1024:.0f} KB")


# ============================================================
# Setup curve: compile kernel, prepare GPU arrays, get kernel attrs
# ============================================================
def setup_curve(cfg):
    """Compile CUDA kernel and prepare GPU arrays for a given curve.
    Returns kernel function, GPU arrays, and kernel attributes dict."""
    script_path = os.path.join(SCRIPT_DIR, cfg['script'])
    kernel_code = extract_kernel_code(script_path)

    cuda_root = "/usr/local/cuda-12.6"
    include_paths = [f"{cuda_root}/include", f"{cuda_root}/include/crt"]
    options = ('--std=c++11',) + tuple(f'-I{p}' for p in include_paths)

    mod = cp.RawModule(code=kernel_code, options=options)
    kernel_fn = mod.get_function(cfg['kernel_name'])

    # Get kernel resource attributes
    kernel_attrs = kernel_fn.attributes
    attrs = {
        'num_regs': kernel_attrs.get('num_regs', 'N/A'),
        'shared_size_bytes': kernel_attrs.get('shared_size_bytes', 0),
        'const_size_bytes': kernel_attrs.get('const_size_bytes', 0),
        'local_size_bytes': kernel_attrs.get('local_size_bytes', 0),
        'max_threads_per_block': kernel_attrs.get('max_threads_per_block', 'N/A'),
        'ptx_version': kernel_attrs.get('ptx_version', 'N/A'),
        'binary_version': kernel_attrs.get('binary_version', 'N/A'),
    }

    n = cfg['n_limbs']
    Px = cp.asarray(to_limbs(cfg['Gx'], n))
    Py = cp.asarray(to_limbs(cfg['Gy'], n))
    Pz = cp.asarray(to_limbs(cfg['Gz'], n))
    a_gpu = cp.asarray(to_limbs(cfg['a'], n))
    b_gpu = cp.asarray(to_limbs(cfg['b'], n))
    Xout = cp.zeros(n, dtype=cp.uint64)
    Yout = cp.zeros(n, dtype=cp.uint64)
    Zout = cp.zeros(n, dtype=cp.uint64)

    return kernel_fn, Px, Py, Pz, a_gpu, b_gpu, Xout, Yout, Zout, attrs


# ============================================================
# Run a single scalar multiplication with timing
# ============================================================
def run_kernel_once(kernel_fn, Px, Py, Pz, a_gpu, b_gpu, Xout, Yout, Zout, cfg):
    """Run one scalar multiplication with a random scalar.
    Returns (cuda_ms, wall_ms, k_int, x_aff_int).
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

    # Read back affine x coordinate
    x_aff = limbs_to_int(Xout, n)

    return cuda_ms, wall_ms, k_int, x_aff


# ============================================================
# Benchmark a single curve
# ============================================================
def benchmark_curve(curve_key, cfg):
    """Run warm-up + timed benchmark for one curve.
    Returns (results_list, kernel_attrs, idle_power_w, mem_after)."""
    print(f"\n{'=' * 64}")
    print(f"  Benchmarking: {cfg['script']}  ({cfg['label']})")
    print(f"{'=' * 64}")

    # Compile kernel and prepare GPU arrays
    print(f"  Compiling CUDA kernel ({cfg['kernel_name']})...")
    kernel_fn, Px, Py, Pz, a_gpu, b_gpu, Xout, Yout, Zout, kernel_attrs = setup_curve(cfg)
    print(f"  Compilation done.")
    print(f"    Registers/Thread:    {kernel_attrs['num_regs']}")
    print(f"    Local Memory/Thread: {kernel_attrs['local_size_bytes']} bytes")
    print(f"    Max Threads/Block:   {kernel_attrs['max_threads_per_block']}")

    # Measure idle power before benchmark
    time.sleep(1)
    idle_power = get_idle_power()
    if idle_power is not None:
        print(f"  Idle GPU Power:        {idle_power:.1f} W")
    else:
        print(f"  Idle GPU Power:        N/A (not available on WSL2)")

    # Warm-up runs (not timed)
    print(f"  Warm-up ({WARMUP_RUNS} runs)...")
    for w in range(WARMUP_RUNS):
        run_kernel_once(kernel_fn, Px, Py, Pz, a_gpu, b_gpu, Xout, Yout, Zout, cfg)
        print(f"    Warm-up {w + 1}/{WARMUP_RUNS} done.")

    # Timed runs
    print(f"  Timed runs ({TIMED_RUNS} runs)...")
    results = []
    for i in range(TIMED_RUNS):
        cuda_ms, wall_ms, k_int, x_aff = run_kernel_once(
            kernel_fn, Px, Py, Pz, a_gpu, b_gpu, Xout, Yout, Zout, cfg
        )

        # GPU stats via nvidia-smi (captured after kernel + sync)
        gpu_stats = get_gpu_stats()

        # Host stats
        cpu_util = psutil.cpu_percent(interval=None)
        ram_used = psutil.virtual_memory().used / (1024 ** 2)

        # Energy (mJ) = power_W * latency_s * 1000
        power_w = gpu_stats.get('power_draw')
        energy_mj = (power_w * (cuda_ms / 1000.0) * 1000.0) if power_w is not None else None

        run_data = {
            'run': i + 1,
            'latency_ms': cuda_ms,
            'wall_ms': wall_ms,
            'k_int': k_int,
            'x_aff': x_aff,
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

        print(f"  Benchmarking {cfg['script']} — Run {i + 1}/{TIMED_RUNS}... "
              f"{cuda_ms:.3f} ms")

    mem_after = get_gpu_memory()
    return results, kernel_attrs, idle_power, mem_after


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
# Write full benchmark report for one curve (Jetson-style)
# ============================================================
def write_stats_file(filename, cfg, results, kernel_attrs, idle_power, dev_props, mem_after):
    """Write full benchmark report to a text file (matching Jetson format)."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = cfg['n_limbs']
    hw = cfg['hex_width']

    latencies = [r['latency_ms'] for r in results]
    wall_times = [r['wall_ms'] for r in results]
    gpu_utils = [r['gpu_util'] for r in results if r['gpu_util'] is not None]
    mem_useds = [r['mem_used'] for r in results if r['mem_used'] is not None]
    temps     = [r['temperature'] for r in results if r['temperature'] is not None]
    powers    = [r['power_draw'] for r in results if r['power_draw'] is not None]
    clocks    = [r['clock_sm'] for r in results if r['clock_sm'] is not None]
    cpu_utils = [r['cpu_util'] for r in results]
    ram_useds = [r['ram_used'] for r in results]
    energies  = [r['energy_mj'] for r in results if r['energy_mj'] is not None]

    lat_arr = np.array(latencies)
    wall_arr = np.array(wall_times)

    W = 75  # output line width
    lines = []
    lines.append("=" * W)
    lines.append("FULL GPU BENCHMARK REPORT")
    lines.append(f"GF(2^{cfg['M']}) Scalar Multiplication - {cfg['label']}")
    lines.append("Lopez-Dahab Projective Coordinates + Itoh-Tsujii Inversion")
    lines.append("=" * W)
    lines.append(f"Date: {now}")
    lines.append(f"Benchmark Runs: {TIMED_RUNS} (+ {WARMUP_RUNS} warmup)")

    # --- 1. Hardware Platform ---
    lines.append("")
    lines.append("-" * W)
    lines.append("1. HARDWARE PLATFORM")
    lines.append("-" * W)
    lines.append(f"  GPU:                       {dev_props['name']}")
    lines.append(f"  Compute Capability:        {dev_props['compute_major']}.{dev_props['compute_minor']}")
    lines.append(f"  Streaming Multiprocessors: {dev_props['sm_count']}")
    lines.append(f"  GPU Clock Rate:            {dev_props['clock_rate_khz']/1000:.0f} MHz")
    lines.append(f"  Memory Clock Rate:         {dev_props['mem_clock_khz']/1000:.0f} MHz")
    lines.append(f"  Total Global Memory:       {dev_props['total_mem'] / (1024**3):.2f} GB")
    lines.append(f"  L2 Cache Size:             {dev_props['l2_cache'] / 1024:.0f} KB")
    lines.append(f"  Warp Size:                 {dev_props['warp_size']}")
    lines.append(f"  Max Threads/Block:         {dev_props['max_threads_per_block']}")
    lines.append(f"  Max Threads/SM:            {dev_props['max_threads_per_sm']}")
    lines.append(f"  Max Blocks/SM:             {dev_props['max_blocks_per_sm']}")
    lines.append(f"  Registers/Block:           {dev_props['regs_per_block']}")
    lines.append(f"  Registers/SM:              {dev_props['regs_per_sm']}")
    lines.append(f"  Shared Memory/Block:       {dev_props['shared_mem_per_block'] / 1024:.1f} KB")
    lines.append(f"  Shared Memory/SM:          {dev_props['shared_mem_per_sm'] / 1024:.1f} KB")
    lines.append(f"  CUDA Version:              {cp.cuda.runtime.runtimeGetVersion()}")
    lines.append(f"  CuPy Version:              {cp.__version__}")

    # --- 2. Kernel Resource Usage ---
    lines.append("")
    lines.append("-" * W)
    lines.append("2. KERNEL RESOURCE USAGE")
    lines.append("-" * W)
    lines.append(f"  Kernel Name:               {cfg['kernel_name']}")
    lines.append(f"  Grid Size:                 (1, 1, 1)")
    lines.append(f"  Block Size:                (1, 1, 1)")
    lines.append(f"  Registers/Thread:          {kernel_attrs['num_regs']}")
    lines.append(f"  Shared Memory (static):    {kernel_attrs['shared_size_bytes']} bytes")
    lines.append(f"  Constant Memory:           {kernel_attrs['const_size_bytes']} bytes")
    lines.append(f"  Local Memory/Thread:       {kernel_attrs['local_size_bytes']} bytes")
    lines.append(f"  Max Threads/Block (kernel):{kernel_attrs['max_threads_per_block']}")
    lines.append(f"  PTX Version:               {kernel_attrs['ptx_version']}")
    lines.append(f"  Binary Version:            {kernel_attrs['binary_version']}")

    # Occupancy calculation
    warp_size = dev_props['warp_size']
    max_warps_per_sm = dev_props['max_threads_per_sm'] // warp_size if warp_size else 0
    active_warps = 1  # single-thread kernel = 1 warp
    occupancy = (active_warps / max_warps_per_sm * 100) if max_warps_per_sm else 0
    lines.append(f"  Theoretical Occupancy:     {occupancy:.2f}% ({active_warps}/{max_warps_per_sm} warps)")
    lines.append(f"  Note: Single-thread kernel - occupancy is intentionally low.")
    lines.append(f"        This kernel performs a single ECC scalar multiplication.")

    # --- 3. GPU Memory Usage ---
    lines.append("")
    lines.append("-" * W)
    lines.append("3. GPU MEMORY USAGE")
    lines.append("-" * W)
    lines.append(f"  CuPy Pool - Used:          {mem_after['used_bytes'] / 1024:.2f} KB")
    lines.append(f"  CuPy Pool - Total Alloc:   {mem_after['total_bytes'] / 1024:.2f} KB")
    buf_bytes = 9 * n * 8
    lines.append(f"  Input Buffers:             9 x {n} x uint64 = {buf_bytes} bytes (Px,Py,Pz,a,b,k,Xout,Yout,Zout)")
    if ram_useds:
        lines.append(f"  System RAM Used (avg):     {np.mean(ram_useds):.0f} MB")

    # --- 4. Latency Measurements ---
    lines.append("")
    lines.append("-" * W)
    lines.append("4. LATENCY MEASUREMENTS")
    lines.append("-" * W)

    for label, arr in [("Wall-clock latency", wall_arr), ("CUDA event latency", lat_arr)]:
        lines.append("")
        lines.append(f"  {label} ({len(arr)} runs):")
        lines.append(f"    Mean:                    {np.mean(arr):.4f} ms")
        lines.append(f"    Median:                  {np.median(arr):.4f} ms")
        lines.append(f"    Std Deviation:           {np.std(arr):.4f} ms")
        lines.append(f"    Min:                     {np.min(arr):.4f} ms")
        lines.append(f"    Max:                     {np.max(arr):.4f} ms")
        lines.append(f"    5th Percentile:          {np.percentile(arr, 5):.4f} ms")
        lines.append(f"    25th Percentile:         {np.percentile(arr, 25):.4f} ms")
        lines.append(f"    75th Percentile:         {np.percentile(arr, 75):.4f} ms")
        lines.append(f"    95th Percentile:         {np.percentile(arr, 95):.4f} ms")
        lines.append(f"    99th Percentile:         {np.percentile(arr, 99):.4f} ms")
        lines.append(f"    IQR:                     {np.percentile(arr, 75) - np.percentile(arr, 25):.4f} ms")
        lines.append(f"    Total:                   {np.sum(arr):.4f} ms")
        lines.append(f"    Throughput:              {1000.0 / np.mean(arr):.2f} ops/sec")

    # Per-run latency detail table
    lines.append("")
    lines.append("  Per-Run Latency Detail:")
    lines.append(f"  {'Run':<6}{'Wall (ms)':<14}{'CUDA (ms)':<14}{'Scalar k (hex)':<{hw+5}}{'x_aff (hex)'}")
    lines.append("  " + "-" * (W - 2))
    for r in results:
        lines.append(f"  {r['run']:<6}{r['wall_ms']:<14.4f}{r['latency_ms']:<14.4f}"
                     f"0x{r['k_int']:0{hw}x}   0x{r['x_aff']:0{hw}x}")

    # --- 5. Power Consumption ---
    lines.append("")
    lines.append("-" * W)
    lines.append("5. POWER CONSUMPTION")
    lines.append("-" * W)
    if idle_power is not None:
        lines.append(f"  Idle GPU Power:            {idle_power*1000:.0f} mW")
    else:
        lines.append(f"  Idle GPU Power:            N/A (WSL2 does not expose power.draw)")

    if powers:
        pw_arr = np.array(powers)
        lines.append(f"")
        lines.append(f"  Under Load (nvidia-smi, {len(pw_arr)} samples):")
        lines.append(f"    Mean Power:              {np.mean(pw_arr)*1000:.0f} mW")
        lines.append(f"    Min Power:               {np.min(pw_arr)*1000:.0f} mW")
        lines.append(f"    Max Power:               {np.max(pw_arr)*1000:.0f} mW")
        lines.append(f"    Std Dev:                 {np.std(pw_arr)*1000:.0f} mW")
    else:
        lines.append(f"")
        lines.append(f"  Under Load:                N/A (WSL2 does not expose power.draw)")
        lines.append(f"  Note: nvidia-smi power.draw is not supported under WSL2.")
        lines.append(f"        GPU power limit is 115 W (TDP).")

    if energies:
        lines.append(f"")
        lines.append(f"  Energy per Scalar Multiplication:")
        avg_pow = np.mean(powers) * 1000
        lines.append(f"    Avg Power:               {avg_pow:.0f} mW")
        lines.append(f"    Avg CUDA Time:           {np.mean(lat_arr):.4f} ms")
        energy_mj = np.mean(energies)
        lines.append(f"    Energy/Op:               {energy_mj:.4f} mJ ({energy_mj*1000:.2f} uJ)")

    # --- 6. Thermal Monitoring ---
    lines.append("")
    lines.append("-" * W)
    lines.append("6. THERMAL MONITORING")
    lines.append("-" * W)
    if temps:
        t_arr = np.array(temps)
        lines.append(f"  GPU Temperature ({len(t_arr)} samples):")
        lines.append(f"    Mean:                    {np.mean(t_arr):.1f} C")
        lines.append(f"    Min:                     {np.min(t_arr):.1f} C")
        lines.append(f"    Max:                     {np.max(t_arr):.1f} C")
    else:
        lines.append(f"  GPU Temperature:           N/A")

    # CPU temperature (try psutil sensors)
    try:
        cpu_temps_dict = psutil.sensors_temperatures()
        if cpu_temps_dict:
            for name, entries in cpu_temps_dict.items():
                if entries:
                    cpu_temp_vals = [e.current for e in entries if e.current > 0]
                    if cpu_temp_vals:
                        lines.append(f"")
                        lines.append(f"  CPU Temperature ({name}, {len(cpu_temp_vals)} sensors):")
                        lines.append(f"    Mean:                    {np.mean(cpu_temp_vals):.1f} C")
                        lines.append(f"    Min:                     {np.min(cpu_temp_vals):.1f} C")
                        lines.append(f"    Max:                     {np.max(cpu_temp_vals):.1f} C")
                        break
    except Exception:
        pass

    # --- 7. GPU Utilization ---
    lines.append("")
    lines.append("-" * W)
    lines.append("7. GPU UTILIZATION")
    lines.append("-" * W)
    if gpu_utils:
        gu_arr = np.array(gpu_utils)
        lines.append(f"  GPU Utilization ({len(gu_arr)} samples):")
        lines.append(f"    Mean:                    {np.mean(gu_arr):.1f}%")
        lines.append(f"    Min:                     {np.min(gu_arr):.0f}%")
        lines.append(f"    Max:                     {np.max(gu_arr):.0f}%")
    else:
        lines.append(f"  GPU Utilization:           N/A")

    if cpu_utils:
        lines.append(f"")
        lines.append(f"  CPU Average Load ({len(cpu_utils)} samples):")
        lines.append(f"    Mean:                    {np.mean(cpu_utils):.1f}%")
        lines.append(f"    Min:                     {np.min(cpu_utils):.1f}%")
        lines.append(f"    Max:                     {np.max(cpu_utils):.1f}%")

    # --- Summary ---
    lines.append("")
    lines.append("=" * W)
    lines.append("SUMMARY")
    lines.append("=" * W)
    lines.append(f"  Curve:                     {cfg['label']} (GF(2^{cfg['M']}))")
    lines.append(f"  Algorithm:                 Lopez-Dahab double-and-add")
    lines.append(f"  Inversion:                 Itoh-Tsujii (addition chain)")
    lines.append(f"  Field Representation:      {n} x 64-bit limbs (polynomial basis)")
    lines.append(f"  GPU:                       {dev_props['name']}")
    lines.append(f"  Avg Latency (CUDA):        {np.mean(lat_arr):.4f} ms")
    lines.append(f"  Avg Latency (Wall):        {np.mean(wall_arr):.4f} ms")
    lines.append(f"  Throughput:                {1000.0 / np.mean(lat_arr):.2f} ops/sec")
    lines.append(f"  Registers/Thread:          {kernel_attrs['num_regs']}")
    if powers:
        lines.append(f"  Avg GPU Power:             {np.mean(powers)*1000:.0f} mW")
    if temps:
        lines.append(f"  Avg GPU Temp:              {np.mean(temps):.1f} C")
    if energies:
        lines.append(f"  Energy/Op:                 {np.mean(energies):.4f} mJ")
    lines.append("=" * W)

    filepath = os.path.join(SCRIPT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Stats written to: {filename}")


# ============================================================
# Write summary file (all curves side-by-side, Jetson-style)
# ============================================================
def write_summary_file(all_results, all_kernel_attrs, all_idle_powers, dev_props):
    """Write a comprehensive summary file with all curves compared."""
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    W = 75

    lines = []
    lines.append("=" * W)
    lines.append("NVIDIA RTX 4060 - GPU BENCHMARK SUMMARY")
    lines.append("ECC Scalar Multiplication over Binary Fields")
    lines.append("=" * W)
    lines.append(f"Date: {now}")
    lines.append(f"Platform: {dev_props['name']} (Compute {dev_props['compute_major']}.{dev_props['compute_minor']}, "
                 f"{dev_props['sm_count']} SMs, {dev_props['clock_rate_khz']/1000:.0f} MHz)")
    lines.append(f"Algorithm: Lopez-Dahab double-and-add (Projective Coordinates)")
    lines.append(f"Inversion: Itoh-Tsujii (addition chain)")
    lines.append(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")
    lines.append(f"CuPy Version: {cp.__version__}")
    lines.append(f"Benchmark Runs: {TIMED_RUNS} (+ {WARMUP_RUNS} warmup) per curve")

    # --- Latency & Throughput ---
    lines.append("")
    lines.append("-" * W)
    lines.append("LATENCY & THROUGHPUT")
    lines.append("-" * W)
    lines.append(f"  {'Curve':<19}{'Field Repr.':<18}{'Avg CUDA (ms)':<16}{'Avg Wall (ms)':<16}{'Throughput'}")
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        results = all_results[key]
        lat = np.array([r['latency_ms'] for r in results])
        wall = np.array([r['wall_ms'] for r in results])
        throughput = 1000.0 / np.mean(lat)
        lines.append(f"  {cfg['label']:<19}{cfg['n_limbs']} x 64-bit{'':<10}"
                     f"{np.mean(lat):<16.4f}{np.mean(wall):<16.4f}{throughput:.2f} ops/sec")

    # --- Kernel Resource Usage ---
    lines.append("")
    lines.append("-" * W)
    lines.append("KERNEL RESOURCE USAGE")
    lines.append("-" * W)
    lines.append(f"  {'Curve':<19}{'Registers/Thread':<18}{'Local Mem/Thread':<18}{'Max Threads/Block'}")
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        attrs = all_kernel_attrs[key]
        lines.append(f"  {cfg['label']:<19}{attrs['num_regs']:<18}"
                     f"{attrs['local_size_bytes']} bytes{'':<10}"
                     f"{attrs['max_threads_per_block']}")

    lines.append("")
    lines.append(f"  Grid Size:         (1, 1, 1) for all curves")
    lines.append(f"  Block Size:        (1, 1, 1) for all curves")
    warp_size = dev_props['warp_size']
    max_warps = dev_props['max_threads_per_sm'] // warp_size if warp_size else 0
    occupancy = (1 / max_warps * 100) if max_warps else 0
    lines.append(f"  Occupancy:         {occupancy:.2f}% (1/{max_warps} warps) - single-thread kernel")

    # --- Power Consumption ---
    lines.append("")
    lines.append("-" * W)
    lines.append("POWER CONSUMPTION")
    lines.append("-" * W)
    # Check if any power data is available
    has_power = any(
        any(r['power_draw'] is not None for r in all_results[key])
        for key in ['163', '233', '571']
    )
    if has_power:
        lines.append(f"  {'Curve':<19}{'Idle (W)':<12}{'Avg Load (W)':<17}{'Energy/Op (mJ)'}")
        for key in ['163', '233', '571']:
            cfg = CURVES[key]
            results = all_results[key]
            idle_w = all_idle_powers[key]
            pws = [r['power_draw'] for r in results if r['power_draw'] is not None]
            ens = [r['energy_mj'] for r in results if r['energy_mj'] is not None]
            idle_str = f"{idle_w:.1f}" if idle_w is not None else "N/A"
            avg_pow_str = f"{np.mean(pws):.1f}" if pws else "N/A"
            avg_energy_str = f"{np.mean(ens):.2f}" if ens else "N/A"
            lines.append(f"  {cfg['label']:<19}{idle_str:<12}{avg_pow_str:<17}{avg_energy_str}")
    else:
        lines.append(f"  Power data not available (WSL2 does not expose nvidia-smi power.draw)")
        lines.append(f"  GPU TDP (power limit): 115 W")

    # --- Thermal Monitoring ---
    lines.append("")
    lines.append("-" * W)
    lines.append("THERMAL MONITORING")
    lines.append("-" * W)
    lines.append(f"  {'Curve':<19}{'Avg GPU Temp (C)':<20}{'Min GPU Temp (C)':<20}{'Max GPU Temp (C)'}")
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        results = all_results[key]
        ts = [r['temperature'] for r in results if r['temperature'] is not None]
        if ts:
            t_arr = np.array(ts)
            lines.append(f"  {cfg['label']:<19}{np.mean(t_arr):<20.1f}{np.min(t_arr):<20.0f}{np.max(t_arr):.0f}")
        else:
            lines.append(f"  {cfg['label']:<19}{'N/A':<20}{'N/A':<20}N/A")

    # --- GPU Utilization ---
    lines.append("")
    lines.append("-" * W)
    lines.append("GPU UTILIZATION")
    lines.append("-" * W)
    lines.append(f"  {'Curve':<19}{'Avg GPU Load (%)':<20}{'Max GPU Load (%)':<20}{'Avg CPU Load (%)'}")
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        results = all_results[key]
        gu = [r['gpu_util'] for r in results if r['gpu_util'] is not None]
        cu = [r['cpu_util'] for r in results]
        if gu:
            gu_arr = np.array(gu)
            lines.append(f"  {cfg['label']:<19}{np.mean(gu_arr):<20.1f}{np.max(gu_arr):<20.0f}"
                         f"{np.mean(cu):.1f}")
        else:
            lines.append(f"  {cfg['label']:<19}{'N/A':<20}{'N/A':<20}{np.mean(cu):.1f}")

    # --- Latency Statistics ---
    lines.append("")
    lines.append("-" * W)
    lines.append("LATENCY STATISTICS (CUDA event, ms)")
    lines.append("-" * W)
    lines.append(f"  {'Curve':<12}{'Mean':<10}{'Median':<10}{'StdDev':<10}{'Min':<10}{'Max':<10}{'P5':<10}{'P95'}")
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        results = all_results[key]
        lat = np.array([r['latency_ms'] for r in results])
        lines.append(f"  {cfg['label']:<12}"
                     f"{np.mean(lat):<10.2f}"
                     f"{np.median(lat):<10.2f}"
                     f"{np.std(lat):<10.2f}"
                     f"{np.min(lat):<10.2f}"
                     f"{np.max(lat):<10.2f}"
                     f"{np.percentile(lat, 5):<10.2f}"
                     f"{np.percentile(lat, 95):.2f}")

    # --- Key Observations ---
    lines.append("")
    lines.append("=" * W)
    lines.append("KEY OBSERVATIONS")
    lines.append("=" * W)

    # Compute ratios
    lat_163 = np.mean([r['latency_ms'] for r in all_results['163']])
    lat_233 = np.mean([r['latency_ms'] for r in all_results['233']])
    lat_571 = np.mean([r['latency_ms'] for r in all_results['571']])
    ratio_571_163 = lat_571 / lat_163 if lat_163 > 0 else 0
    ratio_571_233 = lat_571 / lat_233 if lat_233 > 0 else 0

    lines.append(f"  - GF(2^571) is ~{ratio_571_163:.1f}x slower than GF(2^163) "
                 f"and ~{ratio_571_233:.1f}x slower than GF(2^233)")

    # Energy ratio
    en_163 = [r['energy_mj'] for r in all_results['163'] if r['energy_mj'] is not None]
    en_571 = [r['energy_mj'] for r in all_results['571'] if r['energy_mj'] is not None]
    if en_163 and en_571:
        en_ratio = np.mean(en_571) / np.mean(en_163)
        lines.append(f"  - Energy per operation scales with latency: "
                     f"571-bit uses ~{en_ratio:.1f}x more energy than 163-bit")

    # Power stability
    all_powers = []
    for key in ['163', '233', '571']:
        pws = [r['power_draw'] for r in all_results[key] if r['power_draw'] is not None]
        if pws:
            all_powers.append(np.mean(pws))
    if len(all_powers) >= 2:
        lines.append(f"  - GPU power remains stable across all curves "
                     f"(~{min(all_powers):.0f}-{max(all_powers):.0f} W)")
    elif not all_powers:
        lines.append(f"  - Power data not available on WSL2 (GPU TDP: 115 W)")

    # Temperature
    all_temps = []
    for key in ['163', '233', '571']:
        ts = [r['temperature'] for r in all_results[key] if r['temperature'] is not None]
        if ts:
            all_temps.append(np.mean(ts))
    if len(all_temps) >= 2:
        lines.append(f"  - GPU temperature stays within safe range "
                     f"({min(all_temps):.0f}-{max(all_temps):.0f} C) for all field sizes")

    # GPU utilization trend
    all_utils = []
    for key in ['163', '233', '571']:
        gu = [r['gpu_util'] for r in all_results[key] if r['gpu_util'] is not None]
        if gu:
            all_utils.append(np.mean(gu))
    if len(all_utils) == 3:
        lines.append(f"  - GPU utilization across field sizes: "
                     f"{all_utils[0]:.1f}% -> {all_utils[1]:.1f}% -> {all_utils[2]:.1f}%")

    # Register cap
    regs = [all_kernel_attrs[k]['num_regs'] for k in ['163', '233', '571']
            if all_kernel_attrs[k]['num_regs'] != 'N/A']
    if regs and all(r == 255 for r in regs):
        lines.append(f"  - All kernels hit the 255 register cap, causing register spilling to local memory")

    lines.append("=" * W)

    filepath = os.path.join(SCRIPT_DIR, 'summary_rtx4060.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Summary written to: summary_rtx4060.txt")


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
    ax.set_title('ECC Scalar Multiplication Latency — NVIDIA RTX 4060')
    ax.set_xticks(runs[::5])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'rtx4060_latency_plot.png'), dpi=150)
    plt.close(fig)
    print(f"  Plot saved: rtx4060_latency_plot.png")

    # ---- Plot 2: Power Draw ----
    fig, ax = plt.subplots(figsize=(10, 6))
    has_power = False
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        powers = [r['power_draw'] if r['power_draw'] is not None else 0
                  for r in all_results[key]]
        if any(p > 0 for p in powers):
            has_power = True
        ax.plot(runs, powers, marker='o', color=cfg['color'], label=cfg['label'])
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Power Draw (W)')
    ax.set_title('GPU Power Draw per Run — NVIDIA RTX 4060')
    if not has_power:
        ax.set_title('GPU Power Draw per Run — NVIDIA RTX 4060 (N/A on WSL2)')
    ax.set_xticks(runs[::5])
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
    ax.set_title('GPU Utilization per Run — NVIDIA RTX 4060')
    ax.set_xticks(runs[::5])
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
    print("  ECC GPU Benchmark — NVIDIA RTX 4060")
    print("=" * 64)
    print()

    # Get detailed device properties
    dev_props = get_device_props()

    # Print GPU device info for verification
    print("GPU Device Info:")
    print_gpu_info(dev_props)
    print()

    # Initialize psutil CPU counter (first call always returns 0.0)
    psutil.cpu_percent(interval=None)

    all_results = {}
    all_kernel_attrs = {}
    all_idle_powers = {}
    all_mem_after = {}
    stats_files = {
        '163': 'rtx4060_163_stats.txt',
        '233': 'rtx4060_233_stats.txt',
        '571': 'rtx4060_571_stats.txt',
    }

    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        results, kernel_attrs, idle_power, mem_after = benchmark_curve(key, cfg)
        all_results[key] = results
        all_kernel_attrs[key] = kernel_attrs
        all_idle_powers[key] = idle_power
        all_mem_after[key] = mem_after
        write_stats_file(stats_files[key], cfg, results, kernel_attrs, idle_power, dev_props, mem_after)

    # Generate summary file
    print(f"\n{'=' * 64}")
    print("  Generating summary...")
    print(f"{'=' * 64}")
    write_summary_file(all_results, all_kernel_attrs, all_idle_powers, dev_props)

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
