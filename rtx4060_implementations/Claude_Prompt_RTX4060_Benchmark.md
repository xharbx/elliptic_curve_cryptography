I need you to create a Python benchmarking script called `bench_gpu_rtx4060.py` that benchmarks my ECC scalar multiplication GPU kernel scripts on my NVIDIA RTX 4060.

## My existing scripts

I have 3 Python GPU kernel scripts in the same directory:
- `GPU_kernal_ecc_scalar_163.py` — ECC scalar multiplication on sect163k1 (163-bit curve) using CUDA
- `GPU_kernal_ecc_scalar_233.py` — ECC scalar multiplication on sect233k1 (233-bit curve) using CUDA
- `GPU_kernal_ecc_scalar_571.py` — ECC scalar multiplication on sect571k1 (571-bit curve) using CUDA

Each script has a main function/kernel that performs one ECC scalar multiplication (k × P) on the GPU. You need to look at my scripts first to understand how to import and call the GPU kernel function from each one, and how data is transferred to/from the GPU.

## What bench_gpu_rtx4060.py must do

For EACH of the 3 scripts, do the following:

### Warm-up
- Run the GPU kernel function 2 times (discard results, not timed) to prime GPU caches and JIT compilation

### Timed Runs (10 runs per script)
For each of the 10 runs:
1. Use **CUDA events** for GPU kernel timing if PyCUDA/Numba is used: create start and end events, record start, run kernel, record end, synchronize, compute elapsed time in **milliseconds**. If CUDA events are not feasible with the script's framework, fall back to `time.perf_counter()` around the full GPU call (including any synchronization).
2. Also record **wall-clock time** using `time.perf_counter()` for end-to-end host timing (including memory transfers)
3. After each run completes (and GPU syncs), capture the following via `nvidia-smi`:
   - **GPU Utilization (%)**: `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`
   - **GPU Memory Used (MB)**: `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
   - **GPU Temperature (°C)**: `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits`
   - **GPU Power Draw (W)**: `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits`
   - **GPU Clock Speed (MHz)**: `nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits`
4. Also capture **CPU utilization %** using `psutil.cpu_percent(interval=None)` and **RAM usage (MB)** using `psutil.virtual_memory().used / (1024**2)`
5. Compute **energy per run (mJ)** = power_W × latency_seconds × 1000

### Helper function for nvidia-smi
Create a helper function that runs nvidia-smi with subprocess and parses the output. Handle errors gracefully — if nvidia-smi fails, mark values as N/A. Example:
```python
def get_gpu_stats():
    """Query nvidia-smi for GPU stats. Returns dict."""
    import subprocess
    stats = {}
    queries = {
        'gpu_util': 'utilization.gpu',
        'mem_used': 'memory.used',
        'temperature': 'temperature.gpu',
        'power_draw': 'power.draw',
        'clock_sm': 'clocks.current.sm'
    }
    for key, query in queries.items():
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            stats[key] = float(result.stdout.strip())
        except:
            stats[key] = None
    return stats
```

### Output Stats Files
Write results for each curve to its own text file:
- `rtx4060_163_stats.txt`
- `rtx4060_233_stats.txt`
- `rtx4060_571_stats.txt`

Each file must contain:
```
================================================================
  ECC GPU Benchmark Results — NVIDIA RTX 4060
================================================================
Script:         GPU_kernal_ecc_scalar_163.py
Platform:       NVIDIA RTX 4060 (8GB GDDR6)
Date:           <current datetime>
Warm-up runs:   2
Timed runs:     10
----------------------------------------------------------------
Run  Latency(ms)  GPU_Util(%)  GPU_Mem(MB)  Temp(°C)  Power(W)  Clock(MHz)  CPU_Util(%)  RAM(MB)  Energy(mJ)
  1     XX.XXX       XX.X       XXXX.X       XX       XXX.X      XXXX         XX.X      XXXXX.X    X.XXX
  2     XX.XXX       XX.X       XXXX.X       XX       XXX.X      XXXX         XX.X      XXXXX.X    X.XXX
 ...   (all 10 runs)
----------------------------------------------------------------
Summary Statistics:
  Latency:
    Mean:       XX.XXX ms
    Std Dev:    XX.XXX ms
    Min:        XX.XXX ms
    Max:        XX.XXX ms
    Median:     XX.XXX ms
  GPU Resources:
    Avg GPU Util:     XX.X %
    Avg GPU Memory:   XXXX.X MB
    Avg Temperature:  XX.X °C
    Avg Clock Speed:  XXXX.X MHz
  Power & Energy:
    Avg Power Draw:   XXX.X W
    Avg Energy/Run:   X.XXX mJ
    Total Energy:     X.XXX mJ
  Host:
    Avg CPU Util:     XX.X %
    Avg RAM Used:     XXXXX.X MB
================================================================
```

### Output Plot
Generate a single plot saved as `rtx4060_latency_plot.png`:
- **Type**: Line plot with circle markers
- **X-axis**: Run Number (1–10), integer ticks only
- **Y-axis**: Latency (ms)
- **3 lines**: 163-bit (blue, label="sect163k1"), 233-bit (orange, label="sect233k1"), 571-bit (red, label="sect571k1")
- **Mean lines**: Horizontal dashed line for the mean latency of each curve (same color, alpha=0.5)
- **Title**: "ECC Scalar Multiplication Latency — NVIDIA RTX 4060"
- **Legend**, **grid on**, tight layout
- **DPI**: 150
- **Figure size**: 10×6

### BONUS: Generate a second plot `rtx4060_power_plot.png`:
- **Type**: Line plot with markers
- **X-axis**: Run Number (1–10)
- **Y-axis**: Power Draw (W)
- **3 lines**: one per curve (same colors as above)
- **Title**: "GPU Power Draw per Run — NVIDIA RTX 4060"
- **Legend**, **grid on**, tight layout, DPI 150, Figure size 10×6

### BONUS: Generate a third plot `rtx4060_gpu_util_plot.png`:
- **Type**: Line plot with markers
- **X-axis**: Run Number (1–10)
- **Y-axis**: GPU Utilization (%)
- **3 lines**: one per curve (same colors)
- **Title**: "GPU Utilization per Run — NVIDIA RTX 4060"
- **Legend**, **grid on**, tight layout, DPI 150, Figure size 10×6

## Important requirements
- **READ MY ACTUAL SCRIPTS FIRST** to understand the function names, how to import them, how the kernel is launched, and how to call them properly
- Install any missing packages: `pip install psutil matplotlib numpy`
- Use CUDA events for kernel timing if possible, otherwise `time.perf_counter()` with GPU sync
- Call `psutil.cpu_percent(interval=None)` once BEFORE the benchmark loop to initialize it (first call always returns 0.0)
- Print progress to console as it runs (e.g., "Benchmarking GPU_kernal_ecc_scalar_163 — Run 3/10... 12.345 ms")
- Make sure to **synchronize the GPU** (e.g., `cuda.Context.synchronize()` or `torch.cuda.synchronize()`) before and after timing to get accurate measurements
- Handle nvidia-smi failures gracefully — mark as N/A if it fails
- The script should be self-contained and run with: `python bench_gpu_rtx4060.py`
- Before the benchmark loop starts, print GPU device info (name, compute capability, total memory) for verification
