I need you to create a Python benchmarking script called `bench_cpu.py` that benchmarks my ECC scalar multiplication scripts on my Intel Core i9-14900K CPU.

## My existing scripts

I have 3 Python scripts in the same directory:
- `scalar_mult_163.py` — ECC scalar multiplication on sect163k1 (163-bit curve)
- `scalar_mult_233.py` — ECC scalar multiplication on sect233k1 (233-bit curve)
- `scalar_mult_571.py` — ECC scalar multiplication on sect571k1 (571-bit curve)

Each script has a main function that performs one ECC scalar multiplication (k × P) when called. You need to look at my scripts first to understand how to import and call the scalar multiplication function from each one.

## What bench_cpu.py must do

For EACH of the 3 scripts, do the following:

### Warm-up
- Run the scalar multiplication function 2 times (discard results, not timed)

### Timed Runs (10 runs per script)
For each of the 10 runs:
1. Record start time using `time.perf_counter()`
2. Call the scalar multiplication function
3. Record end time, compute latency in **milliseconds**
4. Capture **CPU utilization %** using `psutil.cpu_percent(interval=None)`
5. Capture **RAM usage in MB** using `psutil.virtual_memory().used / (1024**2)`
6. Try to capture **CPU power draw (W)** using Intel RAPL (`/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj`) — read before and after each run, compute delta. If RAPL is not available, set power to `N/A`
7. Compute **energy per run (mJ)** = power_W × latency_seconds × 1000 (if power is available)

### Output Stats Files
Write results for each curve to its own text file:
- `cpu_i9_163_stats.txt`
- `cpu_i9_233_stats.txt`
- `cpu_i9_571_stats.txt`

Each file must contain:
```
============================================
  ECC CPU Benchmark Results
============================================
Script:       scalar_mult_XXX.py
Platform:     Intel Core i9-14900K
Date:         <current datetime>
Warm-up runs: 2
Timed runs:   10
--------------------------------------------
Run  Latency(ms)  CPU_Util(%)  RAM(MB)  Power(W)  Energy(mJ)
  1      XX.XX       XX.X      XXXX.X    XX.X      X.XXX
  2      XX.XX       XX.X      XXXX.X    XX.X      X.XXX
 ...    (all 10 runs)
--------------------------------------------
Summary Statistics:
  Mean Latency:   XX.XX ms
  Std Dev:        XX.XX ms
  Min:            XX.XX ms
  Max:            XX.XX ms
  Median:         XX.XX ms
  Avg CPU Util:   XX.X %
  Avg RAM:        XXXX.X MB
  Avg Power:      XX.X W (or N/A)
  Avg Energy:     X.XXX mJ (or N/A)
============================================
```

### Output Plot
Generate a single plot saved as `cpu_i9_latency_plot.png`:
- **Type**: Line plot with circle markers
- **X-axis**: Run Number (1–10), integer ticks
- **Y-axis**: Latency (ms)
- **3 lines**: 163-bit (blue, label="sect163k1"), 233-bit (orange, label="sect233k1"), 571-bit (red, label="sect571k1")
- **Mean lines**: Horizontal dashed line for the mean latency of each curve (same color, alpha=0.5)
- **Title**: "ECC Scalar Multiplication Latency — Intel i9-14900K"
- **Legend**, **grid on**, tight layout
- **DPI**: 150
- **Figure size**: 10×6

## Important requirements
- Read my actual scripts first to understand the function names and how to call them
- Install any missing packages: `pip install psutil matplotlib numpy`
- Use `time.perf_counter()` for timing (NOT `time.time()`)
- Call `psutil.cpu_percent(interval=None)` once BEFORE the benchmark loop to initialize it (first call always returns 0)
- Print progress to console as it runs (e.g., "Running scalar_mult_163 — Run 3/10...")
- Handle the case where RAPL is not accessible (permission denied) gracefully — just mark power as N/A
- The script should be self-contained and run with: `python bench_cpu.py`
