# GPU Benchmarking Plan — NVIDIA RTX 4060
## ECC Scalar Multiplication: GPU_kernal_ecc_scalar_163, 233, 571

---

## 1. Overview

| Item | Detail |
|------|--------|
| **Platform** | NVIDIA RTX 4060 (8GB GDDR6, 3072 CUDA cores, 115W TDP) |
| **Scripts** | `GPU_kernal_ecc_scalar_163.py`, `GPU_kernal_ecc_scalar_233.py`, `GPU_kernal_ecc_scalar_571.py` |
| **Runs** | 10 timed runs per script (+ 2 warm-up, discarded) |
| **Metrics** | Latency (ms), GPU Util (%), GPU Mem (MB), Temp (°C), Power (W), Clock (MHz), CPU Util (%), RAM (MB), Energy (mJ) |
| **Outputs** | 3 stats `.txt` files + 3 plots (latency, power, GPU utilization) |

---

## 2. Pre-Benchmark Checklist

- [ ] NVIDIA driver installed: `nvidia-smi` works
- [ ] CUDA toolkit installed: `nvcc --version` works
- [ ] PyCUDA or Numba CUDA installed and functional
- [ ] Python packages: `pip install psutil matplotlib numpy`
- [ ] Lock GPU clocks for consistency:
  ```bash
  sudo nvidia-smi -pm 1
  sudo nvidia-smi --lock-gpu-clocks=2460,2460    # max boost clock for RTX 4060
  ```
- [ ] System idle — close browsers, games, background apps
- [ ] Verify each script runs correctly:
  ```bash
  python GPU_kernal_ecc_scalar_163.py
  python GPU_kernal_ecc_scalar_233.py
  python GPU_kernal_ecc_scalar_571.py
  ```
- [ ] Check GPU is recognized: `nvidia-smi` shows RTX 4060

---

## 3. Metrics Captured Per Run

| Metric | Source | Unit |
|--------|--------|------|
| **Kernel Latency** | CUDA events or perf_counter + sync | ms |
| **GPU Utilization** | nvidia-smi `utilization.gpu` | % |
| **GPU Memory Used** | nvidia-smi `memory.used` | MB |
| **GPU Temperature** | nvidia-smi `temperature.gpu` | °C |
| **GPU Power Draw** | nvidia-smi `power.draw` | W |
| **GPU SM Clock** | nvidia-smi `clocks.current.sm` | MHz |
| **CPU Utilization** | psutil | % |
| **RAM Usage** | psutil | MB |
| **Energy per Run** | power × latency | mJ |

---

## 4. Output Files

After running `bench_gpu_rtx4060.py`, you'll have:

```
project/
├── GPU_kernal_ecc_scalar_163.py      # Your existing script
├── GPU_kernal_ecc_scalar_233.py      # Your existing script
├── GPU_kernal_ecc_scalar_571.py      # Your existing script
├── bench_gpu_rtx4060.py              # Benchmark wrapper (Claude creates this)
├── rtx4060_163_stats.txt             # Output: full stats for 163-bit
├── rtx4060_233_stats.txt             # Output: full stats for 233-bit
├── rtx4060_571_stats.txt             # Output: full stats for 571-bit
├── rtx4060_latency_plot.png          # Output: latency comparison (3 curves)
├── rtx4060_power_plot.png            # Output: power draw per run
└── rtx4060_gpu_util_plot.png         # Output: GPU utilization per run
```

---

## 5. Execution Steps

```bash
# Step 1: Lock GPU clocks
sudo nvidia-smi -pm 1
sudo nvidia-smi --lock-gpu-clocks=2460,2460

# Step 2: Verify scripts work
python GPU_kernal_ecc_scalar_163.py
python GPU_kernal_ecc_scalar_233.py
python GPU_kernal_ecc_scalar_571.py

# Step 3: Run the benchmark
python bench_gpu_rtx4060.py

# Step 4: Check outputs
ls -la rtx4060_*_stats.txt
ls -la rtx4060_*_plot.png

# Step 5: Unlock GPU clocks when done
sudo nvidia-smi --reset-gpu-clocks
```

---

## 6. Post-Benchmark: What to Save

After benchmarking, save these files — you'll need them for the comparative analysis later:
- `rtx4060_163_stats.txt`
- `rtx4060_233_stats.txt`
- `rtx4060_571_stats.txt`
- `rtx4060_latency_plot.png`
- `rtx4060_power_plot.png`
- `rtx4060_gpu_util_plot.png`

These will be combined with Jetson and CPU results for the final cross-platform comparison plots.
