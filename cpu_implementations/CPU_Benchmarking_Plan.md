# CPU Benchmarking Plan — Intel i9-14900K
## ECC Scalar Multiplication: scalar_mult_163, scalar_mult_233, scalar_mult_571

---

## 1. Overview

| Item | Detail |
|------|--------|
| **Platform** | Intel Core i9-14900K |
| **Scripts** | `scalar_mult_163.py`, `scalar_mult_233.py`, `scalar_mult_571.py` |
| **Runs** | 10 timed runs per script (+ 2 warm-up, discarded) |
| **Metrics** | Latency (ms), CPU utilization (%), RAM usage (MB), Power draw (W), Energy (mJ) |
| **Outputs** | 3 stats `.txt` files + 1 latency plot (PNG) |

---

## 2. Pre-Benchmark Checklist

- [ ] Python 3.8+ installed
- [ ] `pip install psutil matplotlib numpy` done
- [ ] CPU governor set to `performance`: `sudo cpupower frequency-set -g performance`
- [ ] System idle — close browsers, background apps
- [ ] Verify each script runs correctly: `python scalar_mult_163.py` (quick sanity check)
- [ ] Note hyperthreading status (enabled/disabled)
- [ ] (Optional) Install `pyRAPL` for power measurement: `pip install pyRAPL`

---

## 3. What the Benchmark Script Will Do

For **each** of the 3 scripts:

1. **Warm-up**: Run the scalar multiplication function 2 times (discard results)
2. **Timed runs (×10)**:
   - Record `time.perf_counter()` before and after the function call → latency in ms
   - Capture CPU utilization % via `psutil.cpu_percent()`
   - Capture RAM usage (MB) via `psutil.virtual_memory()`
   - (Optional) Capture CPU package power via Intel RAPL
   - Compute energy per run: `power_W × latency_s × 1000` → millijoules
3. **Write stats** to `cpu_i9_{curve}_stats.txt`
4. **Plot** Run # vs Latency (ms) for all 3 curves on one chart → `cpu_i9_latency_plot.png`

---

## 4. Output File Format

Each `cpu_i9_{163|233|571}_stats.txt` will contain:

```
============================================
  ECC CPU Benchmark Results
============================================
Script:       scalar_mult_163.py
Platform:     Intel Core i9-14900K
Date:         2026-02-13 14:30:00
Warm-up runs: 2
Timed runs:   10
--------------------------------------------
Run  Latency(ms)  CPU_Util(%)  RAM(MB)  Power(W)  Energy(mJ)
  1      45.23        12.5     1024.3    85.2       3.854
  2      44.98        11.8     1024.5    84.9       3.817
 ...
--------------------------------------------
Summary Statistics:
  Mean Latency:   45.10 ms
  Std Dev:         0.31 ms
  Min:            44.52 ms
  Max:            45.78 ms
  Median:         45.05 ms
  Avg CPU Util:   12.1 %
  Avg RAM:        1024.4 MB
  Avg Power:      85.0 W
  Avg Energy:     3.836 mJ
============================================
```

---

## 5. Plot Specification

**Plot: `cpu_i9_latency_plot.png`**

- **Type**: Line plot with markers
- **X-axis**: Run Number (1–10)
- **Y-axis**: Latency (ms)
- **Lines**: 3 lines (163-bit = blue, 233-bit = orange, 571-bit = red)
- **Extras**: Horizontal dashed line for mean of each curve, legend, grid

---

## 6. Execution Steps

```bash
# Step 1: Set CPU to performance mode
sudo cpupower frequency-set -g performance

# Step 2: Verify scripts work
python scalar_mult_163.py
python scalar_mult_233.py
python scalar_mult_571.py

# Step 3: Run the benchmark
python bench_cpu.py

# Step 4: Check outputs
ls -la cpu_i9_*_stats.txt
ls -la cpu_i9_latency_plot.png
```

---

## 7. Directory Structure (Expected After Benchmarking)

```
project/
├── scalar_mult_163.py          # Your existing script
├── scalar_mult_233.py          # Your existing script
├── scalar_mult_571.py          # Your existing script
├── bench_cpu.py                # Benchmark wrapper (Claude will create this)
├── cpu_i9_163_stats.txt        # Output: stats for 163-bit
├── cpu_i9_233_stats.txt        # Output: stats for 233-bit
├── cpu_i9_571_stats.txt        # Output: stats for 571-bit
└── cpu_i9_latency_plot.png     # Output: latency comparison plot
```
