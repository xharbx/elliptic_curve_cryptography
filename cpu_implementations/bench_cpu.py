#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECC Scalar Multiplication CPU Benchmark
Platform: Intel Core i9-14900K
Benchmarks sect163k1, sect233k1, sect571k1 scalar multiplication.
Generates full Jetson/RTX-4060-style reports with hardware info,
latency statistics, per-run detail, power, thermal, and utilization.

Usage: python bench_cpu.py
Author: Salah Harb
"""

import sys
import os
import io
import time
import random
import datetime
import platform
import statistics
import subprocess

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
# Constants
# ---------------------------------------------------------------------------
WARMUP_RUNS = 5
TIMED_RUNS = 50
PLATFORM_NAME = "Intel Core i9-14900K"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Curve configurations
# ---------------------------------------------------------------------------
CURVES = {
    '163': {
        'bits': 163,
        'label': 'sect163k1',
        'func': scalar_mult_163.ecc_gf2m_163,
        'Gx': scalar_mult_163.Gx,
        'Gy': scalar_mult_163.Gy,
        'M': 163,
        'color': 'blue',
        'hex_width': 41,
        'out_file': 'cpu_i9_163_stats.txt',
    },
    '233': {
        'bits': 233,
        'label': 'sect233k1',
        'func': scalar_mult_233.ecc_gf2m_233,
        'Gx': scalar_mult_233.Gx,
        'Gy': scalar_mult_233.Gy,
        'M': 233,
        'color': 'orange',
        'hex_width': 59,
        'out_file': 'cpu_i9_233_stats.txt',
    },
    '571': {
        'bits': 571,
        'label': 'sect571k1',
        'func': scalar_mult_571.ecc_gf2m_571,
        'Gx': scalar_mult_571.Gx,
        'Gy': scalar_mult_571.Gy,
        'M': 571,
        'color': 'red',
        'hex_width': 143,
        'out_file': 'cpu_i9_571_stats.txt',
    },
}


# ============================================================
# CPU Hardware Info
# ============================================================
def get_cpu_info():
    """Gather CPU hardware information from platform, psutil, and registry."""
    info = {}

    # Basic platform info
    info['processor'] = platform.processor() or "Unknown"
    info['machine'] = platform.machine()
    info['architecture'] = platform.architecture()[0]
    info['os'] = platform.platform()
    info['python_version'] = sys.version.split()[0]
    info['python_impl'] = platform.python_implementation()

    # Core counts
    info['physical_cores'] = psutil.cpu_count(logical=False) or 0
    info['logical_cores'] = psutil.cpu_count(logical=True) or 0

    # CPU frequency
    freq = psutil.cpu_freq()
    if freq:
        info['freq_current'] = freq.current
        info['freq_min'] = freq.min
        info['freq_max'] = freq.max
    else:
        info['freq_current'] = 0
        info['freq_min'] = 0
        info['freq_max'] = 0

    # Total RAM
    mem = psutil.virtual_memory()
    info['total_ram_gb'] = mem.total / (1024 ** 3)

    # Try to get CPU name from registry (Windows) or /proc/cpuinfo (Linux)
    info['cpu_name'] = _get_cpu_name()

    # Try to get cache sizes
    info['l1_cache'] = 'N/A'
    info['l2_cache'] = 'N/A'
    info['l3_cache'] = 'N/A'
    _get_cache_sizes(info)

    return info


def _get_cpu_name():
    """Get the full CPU brand string."""
    if sys.platform == 'win32':
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return name.strip()
        except Exception:
            pass
    else:
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    return platform.processor() or "Unknown"


def _get_cache_sizes(info):
    """Try to get CPU cache sizes from Windows registry or lscpu."""
    if sys.platform == 'win32':
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            # Try to read cache info from CPUID data
            # Windows doesn't reliably store this in registry;
            # use wmic as fallback
            winreg.CloseKey(key)
        except Exception:
            pass

        # Try wmic for cache sizes
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get',
                 'L2CacheSize,L3CacheSize',
                 '/format:list'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith('L2CacheSize=') and line.split('=')[1]:
                    val = line.split('=')[1].strip()
                    if val:
                        info['l2_cache'] = f"{int(val)} KB"
                elif line.startswith('L3CacheSize=') and line.split('=')[1]:
                    val = line.split('=')[1].strip()
                    if val:
                        info['l3_cache'] = f"{int(val)} KB"
        except Exception:
            pass

        # Try PowerShell for more detailed cache info
        try:
            result = subprocess.run(
                ['powershell', '-Command',
                 "Get-CimInstance -ClassName Win32_CacheMemory | "
                 "Select-Object Purpose, InstalledSize | "
                 "Format-Table -AutoSize"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                line_lower = line.lower()
                if 'l1' in line_lower:
                    parts = line.split()
                    for p in parts:
                        try:
                            val = int(p)
                            if val > 0:
                                info['l1_cache'] = f"{val} KB"
                                break
                        except ValueError:
                            continue
        except Exception:
            pass
    else:
        # Linux: try lscpu
        try:
            result = subprocess.run(
                ['lscpu'], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if 'L1d cache' in line:
                    info['l1_cache'] = line.split(':')[1].strip()
                elif 'L2 cache' in line:
                    info['l2_cache'] = line.split(':')[1].strip()
                elif 'L3 cache' in line:
                    info['l3_cache'] = line.split(':')[1].strip()
        except Exception:
            pass


# ============================================================
# RAPL Power Measurement (Linux)
# ============================================================
RAPL_ENERGY_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"


def rapl_available():
    """Check if RAPL energy counter is readable."""
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


# ============================================================
# CPU Temperature (Windows WMI fallback)
# ============================================================
def get_cpu_temperature():
    """Try to get CPU temperature. Returns degrees C or None."""
    # Method 1: psutil sensors (Linux/macOS)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name in ['coretemp', 'k10temp', 'cpu_thermal', 'acpitz']:
                if name in temps:
                    readings = [e.current for e in temps[name] if e.current > 0]
                    if readings:
                        return statistics.mean(readings)
            # Try any available sensor
            for name, entries in temps.items():
                readings = [e.current for e in entries if e.current > 0]
                if readings:
                    return statistics.mean(readings)
    except (AttributeError, Exception):
        pass

    # Method 2: WMI thermal zone (Windows, requires admin)
    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ['powershell', '-Command',
                 "Get-CimInstance -Namespace root/WMI "
                 "-ClassName MSAcpi_ThermalZoneTemperature "
                 "| Select-Object -ExpandProperty CurrentTemperature"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().splitlines():
                try:
                    # WMI returns temperature in tenths of Kelvin
                    val = float(line.strip())
                    celsius = (val / 10.0) - 273.15
                    if 0 < celsius < 150:
                        return celsius
                except ValueError:
                    continue
        except Exception:
            pass

    return None


# ============================================================
# CPU Power (Windows - try LibreHardwareMonitor WMI)
# ============================================================
def get_cpu_power():
    """Try to get CPU package power in Watts. Returns W or None."""
    # Method 1: RAPL (Linux)
    # (handled separately via before/after energy delta)

    # Method 2: LibreHardwareMonitor WMI (Windows)
    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ['powershell', '-Command',
                 "Get-CimInstance -Namespace root/LibreHardwareMonitor "
                 "-ClassName Sensor | Where-Object { $_.SensorType -eq 'Power' "
                 "-and $_.Name -like '*Package*' } | "
                 "Select-Object -ExpandProperty Value"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().splitlines():
                try:
                    val = float(line.strip())
                    if val > 0:
                        return val
                except ValueError:
                    continue
        except Exception:
            pass

        # Method 3: OpenHardwareMonitor WMI
        try:
            result = subprocess.run(
                ['powershell', '-Command',
                 "Get-CimInstance -Namespace root/OpenHardwareMonitor "
                 "-ClassName Sensor | Where-Object { $_.SensorType -eq 'Power' "
                 "-and $_.Name -like '*Package*' } | "
                 "Select-Object -ExpandProperty Value"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().splitlines():
                try:
                    val = float(line.strip())
                    if val > 0:
                        return val
                except ValueError:
                    continue
        except Exception:
            pass

    return None


# ============================================================
# Suppress stdout during scalar multiplication
# ============================================================
def run_scalar_mult_silent(func, k, G):
    """Run scalar multiplication while suppressing all stdout prints."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = func(k, G)
    finally:
        sys.stdout = old_stdout
    return result


# ============================================================
# Benchmark a single curve
# ============================================================
def benchmark_curve(curve_key, cfg, cpu_info, use_rapl):
    """Run warmup + timed benchmark for one curve.
    Returns list of per-run result dicts."""
    M = cfg['M']
    func = cfg['func']
    G = (cfg['Gx'], cfg['Gy'], 1)

    print(f"\n{'=' * 64}")
    print(f"  Benchmarking: {cfg['label']}  (GF(2^{M}))")
    print(f"{'=' * 64}")

    # Measure idle temperature/power before benchmark
    time.sleep(1)
    idle_temp = get_cpu_temperature()
    idle_power = get_cpu_power()
    if idle_temp is not None:
        print(f"  Idle CPU Temp:         {idle_temp:.1f} C")
    else:
        print(f"  Idle CPU Temp:         N/A")
    if idle_power is not None:
        print(f"  Idle CPU Power:        {idle_power:.1f} W")
    else:
        print(f"  Idle CPU Power:        N/A")

    # Warm-up runs (discarded)
    print(f"  Warm-up ({WARMUP_RUNS} runs)...")
    for w in range(WARMUP_RUNS):
        k_warmup = random.getrandbits(M - 1) | (1 << (M - 1))
        run_scalar_mult_silent(func, k_warmup, G)
        print(f"    Warm-up {w + 1}/{WARMUP_RUNS} done.")

    # Initialize psutil CPU counter
    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=True)

    # Timed runs
    print(f"  Timed runs ({TIMED_RUNS} runs)...")
    results = []
    process = psutil.Process()

    for i in range(TIMED_RUNS):
        # Generate random scalar with MSB set
        k_int = random.getrandbits(M - 1) | (1 << (M - 1))

        # Read RAPL before
        rapl_before = read_rapl_uj() if use_rapl else None

        # Temperature before
        temp = get_cpu_temperature()

        # Power reading
        power_w = get_cpu_power()

        # Time the scalar multiplication
        t_start = time.perf_counter()
        x_aff, y_aff = run_scalar_mult_silent(func, k_int, G)
        t_end = time.perf_counter()

        # Read RAPL after
        rapl_after = read_rapl_uj() if use_rapl else None

        latency_s = t_end - t_start
        latency_ms = latency_s * 1000.0

        # CPU utilization
        cpu_util = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(interval=None, percpu=True)

        # Memory
        ram_used_mb = psutil.virtual_memory().used / (1024 ** 2)
        try:
            proc_mem_mb = process.memory_info().rss / (1024 ** 2)
        except Exception:
            proc_mem_mb = 0

        # Compute energy from RAPL if available
        if use_rapl and rapl_before is not None and rapl_after is not None:
            energy_uj = rapl_after - rapl_before
            if energy_uj < 0:
                energy_uj += 2 ** 32  # counter wrap
            rapl_power_w = (energy_uj / 1e6) / latency_s if latency_s > 0 else 0
            energy_mj = rapl_power_w * latency_s * 1000.0
        elif power_w is not None:
            rapl_power_w = power_w
            energy_mj = power_w * latency_s * 1000.0
        else:
            rapl_power_w = None
            energy_mj = None

        run_data = {
            'run': i + 1,
            'latency_ms': latency_ms,
            'k_int': k_int,
            'x_aff': x_aff,
            'cpu_util': cpu_util,
            'per_core': per_core,
            'ram_used': ram_used_mb,
            'proc_mem': proc_mem_mb,
            'temperature': temp,
            'power_w': rapl_power_w,
            'energy_mj': energy_mj,
        }
        results.append(run_data)

        print(f"    Run {i + 1:>2}/{TIMED_RUNS}  {latency_ms:>10.2f} ms  "
              f"k=0x{k_int:0{cfg['hex_width']}x}")

    return results, idle_temp, idle_power


# ============================================================
# Write full benchmark report for one curve
# ============================================================
def write_stats_file(filename, cfg, results, cpu_info, idle_temp, idle_power):
    """Write full benchmark report to text file (matching GPU report format)."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    M = cfg['M']
    hw = cfg['hex_width']

    latencies = [r['latency_ms'] for r in results]
    cpu_utils = [r['cpu_util'] for r in results]
    ram_useds = [r['ram_used'] for r in results]
    proc_mems = [r['proc_mem'] for r in results]
    temps = [r['temperature'] for r in results if r['temperature'] is not None]
    powers = [r['power_w'] for r in results if r['power_w'] is not None]
    energies = [r['energy_mj'] for r in results if r['energy_mj'] is not None]

    lat_arr = np.array(latencies)

    W = 75  # output line width
    lines = []
    lines.append("=" * W)
    lines.append("FULL CPU BENCHMARK REPORT")
    lines.append(f"GF(2^{M}) Scalar Multiplication - {cfg['label']}")
    lines.append("Lopez-Dahab Projective Coordinates + Itoh-Tsujii Inversion")
    lines.append("=" * W)
    lines.append(f"Date: {now}")
    lines.append(f"Benchmark Runs: {TIMED_RUNS} (+ {WARMUP_RUNS} warmup)")

    # --- 1. Hardware Platform ---
    lines.append("")
    lines.append("-" * W)
    lines.append("1. HARDWARE PLATFORM")
    lines.append("-" * W)
    lines.append(f"  CPU:                       {cpu_info['cpu_name']}")
    lines.append(f"  Architecture:              {cpu_info['machine']} ({cpu_info['architecture']})")
    lines.append(f"  Physical Cores:            {cpu_info['physical_cores']}")
    lines.append(f"  Logical Cores:             {cpu_info['logical_cores']}")
    if cpu_info['freq_max'] > 0:
        lines.append(f"  Base Frequency:            {cpu_info['freq_min']:.0f} MHz")
        lines.append(f"  Max Boost Frequency:       {cpu_info['freq_max']:.0f} MHz")
        lines.append(f"  Current Frequency:         {cpu_info['freq_current']:.0f} MHz")
    lines.append(f"  L1 Cache:                  {cpu_info['l1_cache']}")
    lines.append(f"  L2 Cache:                  {cpu_info['l2_cache']}")
    lines.append(f"  L3 Cache:                  {cpu_info['l3_cache']}")
    lines.append(f"  Total System RAM:          {cpu_info['total_ram_gb']:.2f} GB")
    lines.append(f"  OS:                        {cpu_info['os']}")
    lines.append(f"  Python:                    {cpu_info['python_version']} ({cpu_info['python_impl']})")

    # --- 2. Process Resource Usage ---
    lines.append("")
    lines.append("-" * W)
    lines.append("2. PROCESS RESOURCE USAGE")
    lines.append("-" * W)
    lines.append(f"  Implementation:            Pure Python (arbitrary-precision integers)")
    lines.append(f"  Field Representation:      Python int (bignum, unlimited precision)")
    lines.append(f"  Algorithm:                 Lopez-Dahab double-and-add")
    lines.append(f"  Inversion:                 Itoh-Tsujii (addition chain)")
    lines.append(f"  Execution:                 Single-threaded (1 core)")
    avg_proc_mem = np.mean(proc_mems) if proc_mems else 0
    lines.append(f"  Process RSS Memory (avg):  {avg_proc_mem:.1f} MB")

    # --- 3. System Memory Usage ---
    lines.append("")
    lines.append("-" * W)
    lines.append("3. SYSTEM MEMORY USAGE")
    lines.append("-" * W)
    mem = psutil.virtual_memory()
    lines.append(f"  Total System RAM:          {mem.total / (1024**3):.2f} GB")
    lines.append(f"  Available RAM:             {mem.available / (1024**3):.2f} GB")
    lines.append(f"  Used RAM (avg during bench): {np.mean(ram_useds):.0f} MB")
    lines.append(f"  Process RSS (avg):         {avg_proc_mem:.1f} MB")

    # --- 4. Latency Measurements ---
    lines.append("")
    lines.append("-" * W)
    lines.append("4. LATENCY MEASUREMENTS")
    lines.append("-" * W)

    lines.append("")
    lines.append(f"  Wall-clock latency ({len(lat_arr)} runs):")
    lines.append(f"    Mean:                    {np.mean(lat_arr):.4f} ms")
    lines.append(f"    Median:                  {np.median(lat_arr):.4f} ms")
    lines.append(f"    Std Deviation:           {np.std(lat_arr):.4f} ms")
    lines.append(f"    Min:                     {np.min(lat_arr):.4f} ms")
    lines.append(f"    Max:                     {np.max(lat_arr):.4f} ms")
    lines.append(f"    5th Percentile:          {np.percentile(lat_arr, 5):.4f} ms")
    lines.append(f"    25th Percentile:         {np.percentile(lat_arr, 25):.4f} ms")
    lines.append(f"    75th Percentile:         {np.percentile(lat_arr, 75):.4f} ms")
    lines.append(f"    95th Percentile:         {np.percentile(lat_arr, 95):.4f} ms")
    lines.append(f"    99th Percentile:         {np.percentile(lat_arr, 99):.4f} ms")
    iqr = np.percentile(lat_arr, 75) - np.percentile(lat_arr, 25)
    lines.append(f"    IQR:                     {iqr:.4f} ms")
    lines.append(f"    Total:                   {np.sum(lat_arr):.4f} ms")
    throughput = 1000.0 / np.mean(lat_arr)
    lines.append(f"    Throughput:              {throughput:.2f} ops/sec")

    # Per-run detail table
    lines.append("")
    lines.append("  Per-Run Latency Detail:")
    lines.append(f"  {'Run':<6}{'Latency (ms)':<14}{'Scalar k (hex)':<{hw+5}}{'x_aff (hex)'}")
    lines.append("  " + "-" * (W - 2))
    for r in results:
        lines.append(f"  {r['run']:<6}{r['latency_ms']:<14.4f}"
                     f"0x{r['k_int']:0{hw}x}   0x{r['x_aff']:0{hw}x}")

    # --- 5. Power Consumption ---
    lines.append("")
    lines.append("-" * W)
    lines.append("5. POWER CONSUMPTION")
    lines.append("-" * W)

    if idle_power is not None:
        lines.append(f"  Idle CPU Power:            {idle_power:.1f} W")
    else:
        lines.append(f"  Idle CPU Power:            N/A")

    if powers:
        pw_arr = np.array(powers)
        lines.append(f"")
        lines.append(f"  Under Load ({len(pw_arr)} samples):")
        lines.append(f"    Mean Power:              {np.mean(pw_arr):.1f} W")
        lines.append(f"    Min Power:               {np.min(pw_arr):.1f} W")
        lines.append(f"    Max Power:               {np.max(pw_arr):.1f} W")
        lines.append(f"    Std Dev:                 {np.std(pw_arr):.1f} W")
    else:
        lines.append(f"")
        lines.append(f"  Under Load:                N/A")
        lines.append(f"  Note: CPU power measurement requires RAPL (Linux) or")
        lines.append(f"        LibreHardwareMonitor/OpenHardwareMonitor (Windows).")
        lines.append(f"        i9-14900K TDP: 125 W (PBP) / 253 W (MTP)")

    if energies:
        lines.append(f"")
        lines.append(f"  Energy per Scalar Multiplication:")
        lines.append(f"    Avg Power:               {np.mean(powers):.1f} W")
        lines.append(f"    Avg Latency:             {np.mean(lat_arr):.4f} ms")
        energy_mj = np.mean(energies)
        lines.append(f"    Energy/Op:               {energy_mj:.4f} mJ ({energy_mj*1000:.2f} uJ)")

    # --- 6. Thermal Monitoring ---
    lines.append("")
    lines.append("-" * W)
    lines.append("6. THERMAL MONITORING")
    lines.append("-" * W)
    if temps:
        t_arr = np.array(temps)
        lines.append(f"  CPU Temperature ({len(t_arr)} samples):")
        lines.append(f"    Mean:                    {np.mean(t_arr):.1f} C")
        lines.append(f"    Min:                     {np.min(t_arr):.1f} C")
        lines.append(f"    Max:                     {np.max(t_arr):.1f} C")
    else:
        lines.append(f"  CPU Temperature:           N/A")
        lines.append(f"  Note: Temperature reading requires admin privileges on Windows")
        lines.append(f"        (MSAcpi_ThermalZoneTemperature) or psutil sensor support.")

    if idle_temp is not None:
        lines.append(f"")
        lines.append(f"  Idle Temperature:          {idle_temp:.1f} C")

    # --- 7. CPU Utilization ---
    lines.append("")
    lines.append("-" * W)
    lines.append("7. CPU UTILIZATION")
    lines.append("-" * W)
    if cpu_utils:
        cu_arr = np.array(cpu_utils)
        lines.append(f"  Overall CPU Utilization ({len(cu_arr)} samples):")
        lines.append(f"    Mean:                    {np.mean(cu_arr):.1f}%")
        lines.append(f"    Min:                     {np.min(cu_arr):.1f}%")
        lines.append(f"    Max:                     {np.max(cu_arr):.1f}%")

    # Per-core average utilization
    all_per_core = [r['per_core'] for r in results if r['per_core']]
    if all_per_core:
        n_cores = len(all_per_core[0])
        avg_per_core = []
        for c in range(n_cores):
            core_vals = [pc[c] for pc in all_per_core if c < len(pc)]
            avg_per_core.append(np.mean(core_vals))

        lines.append(f"")
        lines.append(f"  Per-Core Average Utilization ({n_cores} logical cores):")
        # Show top 8 busiest cores + summary
        sorted_cores = sorted(enumerate(avg_per_core), key=lambda x: x[1], reverse=True)
        shown = min(8, len(sorted_cores))
        for idx, avg_val in sorted_cores[:shown]:
            lines.append(f"    Core {idx:>2}:                  {avg_val:.1f}%")
        if len(sorted_cores) > shown:
            remaining = [v for _, v in sorted_cores[shown:]]
            lines.append(f"    (remaining {len(remaining)} cores avg: {np.mean(remaining):.1f}%)")

    lines.append(f"")
    lines.append(f"  System RAM Used (avg):     {np.mean(ram_useds):.0f} MB")

    # --- Summary ---
    lines.append("")
    lines.append("=" * W)
    lines.append("SUMMARY")
    lines.append("=" * W)
    lines.append(f"  Curve:                     {cfg['label']} (GF(2^{M}))")
    lines.append(f"  Algorithm:                 Lopez-Dahab double-and-add")
    lines.append(f"  Inversion:                 Itoh-Tsujii (addition chain)")
    lines.append(f"  Field Representation:      Python int (arbitrary precision)")
    lines.append(f"  CPU:                       {cpu_info['cpu_name']}")
    lines.append(f"  Avg Latency (Wall):        {np.mean(lat_arr):.4f} ms")
    lines.append(f"  Throughput:                {throughput:.2f} ops/sec")
    if powers:
        lines.append(f"  Avg CPU Power:             {np.mean(powers):.1f} W")
    if temps:
        lines.append(f"  Avg CPU Temp:              {np.mean(temps):.1f} C")
    if energies:
        lines.append(f"  Energy/Op:                 {np.mean(energies):.4f} mJ")
    lines.append("=" * W)

    filepath = os.path.join(SCRIPT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  -> Stats written to: {filename}")


# ============================================================
# Write summary file (all curves side-by-side)
# ============================================================
def write_summary_file(all_results, cpu_info):
    """Write a comprehensive summary file with all curves compared."""
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    W = 75

    lines = []
    lines.append("=" * W)
    lines.append(f"Intel Core i9-14900K - CPU BENCHMARK SUMMARY")
    lines.append("ECC Scalar Multiplication over Binary Fields")
    lines.append("=" * W)
    lines.append(f"Date: {now}")
    lines.append(f"Platform: {cpu_info['cpu_name']} "
                 f"({cpu_info['physical_cores']}P+E cores, "
                 f"{cpu_info['logical_cores']} threads)")
    lines.append(f"Algorithm: Lopez-Dahab double-and-add (Projective Coordinates)")
    lines.append(f"Inversion: Itoh-Tsujii (addition chain)")
    lines.append(f"Implementation: Pure Python (arbitrary-precision integers)")
    lines.append(f"Python: {cpu_info['python_version']} ({cpu_info['python_impl']})")
    lines.append(f"OS: {cpu_info['os']}")
    lines.append(f"Benchmark Runs: {TIMED_RUNS} (+ {WARMUP_RUNS} warmup) per curve")

    # --- Latency & Throughput ---
    lines.append("")
    lines.append("-" * W)
    lines.append("LATENCY & THROUGHPUT")
    lines.append("-" * W)
    lines.append(f"  {'Curve':<19}{'Field Repr.':<18}{'Avg Wall (ms)':<16}{'Throughput'}")
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        results = all_results[key]
        lat = np.array([r['latency_ms'] for r in results])
        throughput = 1000.0 / np.mean(lat)
        lines.append(f"  {cfg['label']:<19}{'Python bigint':<18}"
                     f"{np.mean(lat):<16.4f}{throughput:.2f} ops/sec")

    # --- Power Consumption ---
    lines.append("")
    lines.append("-" * W)
    lines.append("POWER CONSUMPTION")
    lines.append("-" * W)
    has_power = any(
        any(r['power_w'] is not None for r in all_results[key])
        for key in ['163', '233', '571']
    )
    if has_power:
        lines.append(f"  {'Curve':<19}{'Avg Power (W)':<17}{'Energy/Op (mJ)'}")
        for key in ['163', '233', '571']:
            cfg = CURVES[key]
            results = all_results[key]
            pws = [r['power_w'] for r in results if r['power_w'] is not None]
            ens = [r['energy_mj'] for r in results if r['energy_mj'] is not None]
            avg_pow_str = f"{np.mean(pws):.1f}" if pws else "N/A"
            avg_energy_str = f"{np.mean(ens):.2f}" if ens else "N/A"
            lines.append(f"  {cfg['label']:<19}{avg_pow_str:<17}{avg_energy_str}")
    else:
        lines.append(f"  Power data not available on this platform.")
        lines.append(f"  CPU TDP: 125 W (PBP) / 253 W (MTP)")

    # --- Thermal Monitoring ---
    lines.append("")
    lines.append("-" * W)
    lines.append("THERMAL MONITORING")
    lines.append("-" * W)
    has_temp = any(
        any(r['temperature'] is not None for r in all_results[key])
        for key in ['163', '233', '571']
    )
    if has_temp:
        lines.append(f"  {'Curve':<19}{'Avg CPU Temp (C)':<20}{'Min CPU Temp (C)':<20}{'Max CPU Temp (C)'}")
        for key in ['163', '233', '571']:
            cfg = CURVES[key]
            results = all_results[key]
            ts = [r['temperature'] for r in results if r['temperature'] is not None]
            if ts:
                t_arr = np.array(ts)
                lines.append(f"  {cfg['label']:<19}{np.mean(t_arr):<20.1f}"
                             f"{np.min(t_arr):<20.1f}{np.max(t_arr):.1f}")
            else:
                lines.append(f"  {cfg['label']:<19}{'N/A':<20}{'N/A':<20}N/A")
    else:
        lines.append(f"  Temperature data not available on this platform.")
        lines.append(f"  Requires admin privileges or hardware monitor software.")

    # --- CPU Utilization ---
    lines.append("")
    lines.append("-" * W)
    lines.append("CPU UTILIZATION")
    lines.append("-" * W)
    lines.append(f"  {'Curve':<19}{'Avg CPU Load (%)':<20}{'Max CPU Load (%)':<20}{'Avg RAM (MB)'}")
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        results = all_results[key]
        cu = [r['cpu_util'] for r in results]
        ram = [r['ram_used'] for r in results]
        lines.append(f"  {cfg['label']:<19}{np.mean(cu):<20.1f}"
                     f"{np.max(cu):<20.1f}{np.mean(ram):.0f}")

    # --- Latency Statistics ---
    lines.append("")
    lines.append("-" * W)
    lines.append("LATENCY STATISTICS (Wall-clock, ms)")
    lines.append("-" * W)
    lines.append(f"  {'Curve':<12}{'Mean':<10}{'Median':<10}{'StdDev':<10}"
                 f"{'Min':<10}{'Max':<10}{'P5':<10}{'P95'}")
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

    lat_163 = np.mean([r['latency_ms'] for r in all_results['163']])
    lat_233 = np.mean([r['latency_ms'] for r in all_results['233']])
    lat_571 = np.mean([r['latency_ms'] for r in all_results['571']])
    ratio_571_163 = lat_571 / lat_163 if lat_163 > 0 else 0
    ratio_571_233 = lat_571 / lat_233 if lat_233 > 0 else 0
    ratio_233_163 = lat_233 / lat_163 if lat_163 > 0 else 0

    lines.append(f"  - GF(2^571) is ~{ratio_571_163:.1f}x slower than GF(2^163) "
                 f"and ~{ratio_571_233:.1f}x slower than GF(2^233)")
    lines.append(f"  - GF(2^233) is ~{ratio_233_163:.1f}x slower than GF(2^163)")

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
        pws = [r['power_w'] for r in all_results[key] if r['power_w'] is not None]
        if pws:
            all_powers.append(np.mean(pws))
    if len(all_powers) >= 2:
        lines.append(f"  - CPU power across curves: "
                     f"~{min(all_powers):.0f}-{max(all_powers):.0f} W")
    elif not all_powers:
        lines.append(f"  - Power data not available (CPU TDP: 125W PBP / 253W MTP)")

    # Temperature
    all_temps = []
    for key in ['163', '233', '571']:
        ts = [r['temperature'] for r in all_results[key] if r['temperature'] is not None]
        if ts:
            all_temps.append(np.mean(ts))
    if len(all_temps) >= 2:
        lines.append(f"  - CPU temperature range: "
                     f"{min(all_temps):.0f}-{max(all_temps):.0f} C across all field sizes")

    # CPU utilization
    all_utils = []
    for key in ['163', '233', '571']:
        cu = [r['cpu_util'] for r in all_results[key]]
        all_utils.append(np.mean(cu))
    if len(all_utils) == 3:
        lines.append(f"  - CPU utilization across field sizes: "
                     f"{all_utils[0]:.1f}% -> {all_utils[1]:.1f}% -> {all_utils[2]:.1f}%")

    lines.append(f"  - Single-threaded Python execution (1 of {cpu_info['logical_cores']} logical cores)")
    lines.append(f"  - Pure Python bigint arithmetic (no C extensions or SIMD)")

    lines.append("=" * W)

    filepath = os.path.join(SCRIPT_DIR, 'summary_i9_cpu.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  -> Summary written to: summary_i9_cpu.txt")


# ============================================================
# Generate plots
# ============================================================
def generate_plots(all_results):
    """Generate latency, power, and CPU utilization plots."""
    runs = list(range(1, TIMED_RUNS + 1))

    # --- Plot 1: Latency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        latencies = [r['latency_ms'] for r in all_results[key]]
        mean_lat = statistics.mean(latencies)
        ax.plot(runs, latencies, marker='o', markersize=3,
                color=cfg['color'], label=cfg['label'])
        ax.axhline(y=mean_lat, color=cfg['color'], linestyle='--', alpha=0.5,
                   label=f"{cfg['label']} mean ({mean_lat:.2f} ms)")
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'ECC Scalar Multiplication Latency \u2014 {PLATFORM_NAME}')
    ax.set_xticks(runs[::5])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'cpu_i9_latency_plot.png'), dpi=150)
    plt.close(fig)
    print(f"  -> Plot saved: cpu_i9_latency_plot.png")

    # --- Plot 2: Power (if available) ---
    has_power = any(
        any(r['power_w'] is not None for r in all_results[key])
        for key in ['163', '233', '571']
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        powers = [r['power_w'] if r['power_w'] is not None else 0
                  for r in all_results[key]]
        ax.plot(runs, powers, marker='o', markersize=3,
                color=cfg['color'], label=cfg['label'])
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Power (W)')
    title = f'CPU Power per Run \u2014 {PLATFORM_NAME}'
    if not has_power:
        title += ' (N/A)'
    ax.set_title(title)
    ax.set_xticks(runs[::5])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'cpu_i9_power_plot.png'), dpi=150)
    plt.close(fig)
    print(f"  -> Plot saved: cpu_i9_power_plot.png")

    # --- Plot 3: CPU Utilization ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        utils = [r['cpu_util'] for r in all_results[key]]
        ax.plot(runs, utils, marker='o', markersize=3,
                color=cfg['color'], label=cfg['label'])
    ax.set_xlabel('Run Number')
    ax.set_ylabel('CPU Utilization (%)')
    ax.set_title(f'CPU Utilization per Run \u2014 {PLATFORM_NAME}')
    ax.set_xticks(runs[::5])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'cpu_i9_cpu_util_plot.png'), dpi=150)
    plt.close(fig)
    print(f"  -> Plot saved: cpu_i9_cpu_util_plot.png")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 64)
    print(f"  ECC CPU Benchmark \u2014 {PLATFORM_NAME}")
    print("=" * 64)
    print()

    # Gather CPU info
    print("Gathering CPU hardware info...")
    cpu_info = get_cpu_info()
    print(f"  CPU:              {cpu_info['cpu_name']}")
    print(f"  Cores:            {cpu_info['physical_cores']} physical, "
          f"{cpu_info['logical_cores']} logical")
    if cpu_info['freq_max'] > 0:
        print(f"  Frequency:        {cpu_info['freq_current']:.0f} MHz "
              f"(max {cpu_info['freq_max']:.0f} MHz)")
    print(f"  RAM:              {cpu_info['total_ram_gb']:.1f} GB")
    print(f"  L2 Cache:         {cpu_info['l2_cache']}")
    print(f"  L3 Cache:         {cpu_info['l3_cache']}")
    print(f"  OS:               {cpu_info['os']}")
    print(f"  Python:           {cpu_info['python_version']}")
    print()

    # Check RAPL availability
    use_rapl = rapl_available()
    if use_rapl:
        print("[INFO] RAPL power measurement available.")
    else:
        print("[INFO] RAPL not available. Trying alternative power sources...")
        test_power = get_cpu_power()
        if test_power is not None:
            print(f"[INFO] Alternative power source found: {test_power:.1f} W")
        else:
            print("[INFO] No power measurement available. Power will be N/A.")

    test_temp = get_cpu_temperature()
    if test_temp is not None:
        print(f"[INFO] CPU temperature available: {test_temp:.1f} C")
    else:
        print("[INFO] CPU temperature not available.")
    print()

    # Initialize psutil CPU counter (first call always returns 0.0)
    psutil.cpu_percent(interval=None)

    all_results = {}
    all_idle_temps = {}
    all_idle_powers = {}

    for key in ['163', '233', '571']:
        cfg = CURVES[key]
        results, idle_temp, idle_power = benchmark_curve(
            key, cfg, cpu_info, use_rapl
        )
        all_results[key] = results
        all_idle_temps[key] = idle_temp
        all_idle_powers[key] = idle_power

        # Write per-curve report
        write_stats_file(cfg['out_file'], cfg, results, cpu_info,
                         idle_temp, idle_power)

    # Write summary file
    print(f"\n{'=' * 64}")
    print("  Generating summary...")
    print(f"{'=' * 64}")
    write_summary_file(all_results, cpu_info)

    # Generate plots
    print(f"\n{'=' * 64}")
    print("  Generating plots...")
    print(f"{'=' * 64}")
    generate_plots(all_results)

    print(f"\n{'=' * 64}")
    print("  Benchmark complete!")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
