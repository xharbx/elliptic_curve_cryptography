# -*- coding: utf-8 -*-
"""
Full GPU Benchmark: GF(2^233) Scalar Multiplication on Jetson Orin Nano Super
Measures: Latency, Hardware Resources, Power Consumption
Author: Salah Harb
"""

import cupy as cp
import numpy as np
import time
import os
import subprocess
import threading
import re
import random
from datetime import datetime

# ============================================================
# Parameters for sect233r1
# ============================================================
M = 233
IRRED_POLY = (1 << 233) | (1 << 74) | 1

a_curve_int = 1
b_curve_int = 0x066647ede6c332c7f8c0923bb58213b333b20e9ce4281fe115f7d8f90ad

Gx_int = 0x0fac9dfcbac8313bb2139f1bb755fef65bc391f8b36f8f8eb7371fd558b
Gy_int = 0x1006a08a41903350678e58528bebf8a0beff867a7ca36716f7e01f81052
Gz_int = 1

def to_limbs(x):
    return np.array([
        np.uint64(x & ((1 << 64) - 1)),
        np.uint64((x >> 64) & ((1 << 64) - 1)),
        np.uint64((x >> 128) & ((1 << 64) - 1)),
        np.uint64(x >> 192)
    ], dtype=np.uint64)

Gx = to_limbs(Gx_int)
Gy = to_limbs(Gy_int)
Gz = to_limbs(Gz_int)
a_curve = to_limbs(a_curve_int)
b_curve = to_limbs(b_curve_int)

# ============================================================
# GPU Kernel Code (from GPU_kernal_ecc_scalar_233.py, silent)
# ============================================================
kernel_code = r'''
#ifndef __UINT64_T_DEFINED
typedef unsigned long long uint64_t;
#define __UINT64_T_DEFINED
#endif

extern "C" {

// ============================================================
// GF(2) bitwise multiply for <=64-bit chunks
// ============================================================
__device__ __forceinline__ void gf2_mul64(uint64_t *hi, uint64_t *lo,
                                          uint64_t a, uint64_t b) {
    *hi = 0; *lo = 0;
    for (int i = 0; i < 64; ++i) {
        if ((b >> i) & 1ULL) {
            *lo ^= (a << i);
            if (i) *hi ^= (a >> (64 - i));
        }
    }
}

// ============================================================
// XOR combine t[] <- t[] ^ (val << shift)
// ============================================================
__device__ void xor_shift(uint64_t *t, const uint64_t *val,
                          int nval, int shift) {
    int w = shift / 64;
    int b = shift % 64;
    for (int i = 0; i < nval; i++) {
        t[w + i] ^= val[i] << b;
        if (b && (w + i + 1 < 8))
            t[w + i + 1] ^= val[i] >> (64 - b);
    }
}


// ============================================================
// Karatsuba 233-bit carryless multiply (4x4 limb)
// ============================================================
__device__ void gf2_mult_233(uint64_t *r, const uint64_t *a,
                             const uint64_t *b) {
    uint64_t t[8] = {0};

    // 4x4 schoolbook — safe baseline for correctness
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            gf2_mul64(&hi, &lo, a[i], b[j]);
            xor_shift(t, &lo, 1, (i + j) * 64);
            xor_shift(t, &hi, 1, (i + j) * 64 + 64);
        }
    }

    // Modular reduction mod x^233 + x^74 + 1
    for (int bit = 511; bit >= 233; --bit) {
        int limb = bit / 64;
        int pos = bit % 64;
        if (t[limb] & (1ULL << pos)) {
            t[limb] ^= (1ULL << pos);
            int shift = bit - 233;
            int targets[2] = {shift + 74, shift};
            for (int k = 0; k < 2; k++) {
                int tgt = targets[k];
                if (tgt < 0) continue;
                int w = tgt / 64, bpos = tgt % 64;
                t[w] ^= (1ULL << bpos);
            }
        }
    }

    r[0] = t[0];
    r[1] = t[1];
    r[2] = t[2];
    r[3] = t[3] & ((1ULL << 41) - 1);
}

// ============================================================
// Squaring
// ============================================================
__device__ void gf2_square(uint64_t *r, const uint64_t *a) {
    uint64_t t[8] = {0};

    for (int limb = 0; limb < 4; limb++) {
        uint64_t x = a[limb];
        for (int bit = 0; bit < 64; bit++) {
            if (x & (1ULL << bit)) {
                int pos = 2 * (bit + 64 * limb);
                int w = pos / 64;
                int bitpos = pos % 64;
                t[w] ^= (1ULL << bitpos);
            }
        }
    }

    // Reduction mod x^233 + x^74 + 1
    for (int bit = 511; bit >= 233; --bit) {
        int limb = bit / 64;
        int pos  = bit % 64;
        if (t[limb] & (1ULL << pos)) {
            t[limb] ^= (1ULL << pos);
            int shift = bit - 233;
            int targets[2] = {shift + 74, shift};
            for (int k = 0; k < 2; k++) {
                int tgt = targets[k];
                if (tgt < 0) continue;
                int w = tgt / 64, bpos = tgt % 64;
                t[w] ^= (1ULL << bpos);
            }
        }
    }

    r[0] = t[0];
    r[1] = t[1];
    r[2] = t[2];
    r[3] = t[3] & ((1ULL << 41) - 1);
}


__device__ void gf2_square_k(uint64_t *r,const uint64_t *a,int k){
    uint64_t t1[4],t2[4];
    for(int i=0;i<4;i++)t1[i]=a[i];
    for(int i=0;i<k;i++){
        gf2_square(t2,t1);
        for(int j=0;j<4;j++)t1[j]=t2[j];
    }
    for(int i=0;i<4;i++)r[i]=t1[i];
}


__device__ bool get_bit(const uint64_t *k, int i) {
    int limb = i / 64;
    int bit  = i % 64;
    return (k[limb] >> bit) & 1ULL;
}

// ============================================================
// Lopez-Dahab Point Double
// ============================================================

__device__ void point_double_233(
    uint64_t *X2, uint64_t *Y2, uint64_t *Z2,
    const uint64_t *X1, const uint64_t *Y1, const uint64_t *Z1,
    const uint64_t *a_curve, const uint64_t *b_curve)
{
    uint64_t X1_sq[4], Z1_sq[4], Z1_4[4], X1_4[4], Y1_sq[4];
    uint64_t bZ1_4[4], aZ2[4], inner[4], right[4];
    uint64_t X2right[4], left[4];

    // Step 1-5: squares
    gf2_square(X1_sq, X1);
    gf2_square(Z1_sq, Z1);
    gf2_square(Z1_4, Z1_sq);
    gf2_square(X1_4, X1_sq);
    gf2_square(Y1_sq, Y1);

    // Step 6: Z2 = X1^2 * Z1^2
    gf2_mult_233(Z2, X1_sq, Z1_sq);

    // Step 7: bZ1_4 = b * Z1^4
    gf2_mult_233(bZ1_4, b_curve, Z1_4);

    // Step 8: X2 = X1^4 + bZ1_4
    for (int i = 0; i < 4; i++) X2[i] = X1_4[i] ^ bZ1_4[i];

    // Step 9: left = (bZ1_4 * Z2)
    gf2_mult_233(left, bZ1_4, Z2);

    // Step 10: aZ2 = a * Z2
    gf2_mult_233(aZ2, a_curve, Z2);

    // Step 11: inner = aZ2 + Y1^2
    for (int i = 0; i < 4; i++) inner[i] = aZ2[i] ^ Y1_sq[i];

    // Step 12: right = inner + bZ1_4
    for (int i = 0; i < 4; i++) right[i] = inner[i] ^ bZ1_4[i];

    // Step 13: X2right = X2 * right
    gf2_mult_233(X2right, X2, right);

    // Step 14: Y2 = left + X2right
    for (int i = 0; i < 4; i++) Y2[i] = left[i] ^ X2right[i];
}

// ============================================================
// Lopez-Dahab Point Add (Z1=1)
// ============================================================
__device__ void point_add_233(uint64_t *X2, uint64_t *Y2, uint64_t *Z2,
                              const uint64_t *X0, const uint64_t *Y0, const uint64_t *Z0,
                              const uint64_t *X1, const uint64_t *Y1,
                              const uint64_t *a_curve) {
    uint64_t T1[4], T2[4], A[4], T3[4], B[4], C[4];
    uint64_t B2[4], T4[4], T5[4], D[4], Z2t[4], E[4], A2[4], T6[4], F[4], G[4];
    uint64_t T7[4], T8[4], T9[4], T10[4];

    // Step 1: A = Y1 * Z0^2 + Y0
    gf2_square(T1, Z0);
    gf2_mult_233(T2, Y1, T1);
    for (int i = 0; i < 4; i++) A[i] = T2[i] ^ Y0[i];

    // Step 2: B = X1 * Z0 + X0
    gf2_mult_233(T3, X1, Z0);
    for (int i = 0; i < 4; i++) B[i] = T3[i] ^ X0[i];

    // Step 3: C = Z0 * B
    gf2_mult_233(C, Z0, B);

    // Step 4: D = B^2 * (C + a * Z0^2)
    gf2_square(B2, B);
    gf2_mult_233(T4, a_curve, T1);
    for (int i = 0; i < 4; i++) T5[i] = C[i] ^ T4[i];
    gf2_mult_233(D, B2, T5);

    // Step 5: Z2 = C^2
    gf2_square(Z2t, C);

    // Step 6: E = A * C
    gf2_mult_233(E, A, C);

    // Step 7: X2 = A^2 + D + E
    gf2_square(A2, A);
    for (int i = 0; i < 4; i++) T6[i] = A2[i] ^ D[i];
    for (int i = 0; i < 4; i++) X2[i] = T6[i] ^ E[i];

    // Step 8: F = X2 + X1 * Z2
    gf2_mult_233(T7, X1, Z2t);
    for (int i = 0; i < 4; i++) F[i] = X2[i] ^ T7[i];

    // Step 9: G = X2 + Y1 * Z2
    gf2_mult_233(T8, Y1, Z2t);
    for (int i = 0; i < 4; i++) G[i] = X2[i] ^ T8[i];

    // Step 10: Y2 = E*F + Z2*G
    gf2_mult_233(T9, E, F);
    gf2_mult_233(T10, Z2t, G);
    for (int i = 0; i < 4; i++) Y2[i] = T9[i] ^ T10[i];

    // Step 11: Copy Z2t -> Z2
    for (int i = 0; i < 4; i++) Z2[i] = Z2t[i];
}


// ============================================================
// Itoh-Tsujii inversion mu-sequence for GF(2^233)
// ============================================================
__device__ void gf2_inv_itoh_tsujii_233(uint64_t *r, const uint64_t *a) {
    uint64_t t1[4], t2[4], t3[4];
    uint64_t b1[4] = {a[0], a[1], a[2], a[3]};
    uint64_t b2[4], b4[4], b8[4], b16[4], b32[4], b64[4], b128[4], b192[4], b224[4], b232[4];

    // b2 = (b1)^2 * b1
    gf2_square(t1, b1);
    gf2_mult_233(b2, t1, b1);

    // b4 = (b2)^(2^2) * b2
    gf2_square_k(t1, b2, 2);
    gf2_mult_233(b4, t1, b2);

    // b8 = (b4)^(2^4) * b4
    gf2_square_k(t1, b4, 4);
    gf2_mult_233(b8, t1, b4);

    // b16 = (b8)^(2^8) * b8
    gf2_square_k(t1, b8, 8);
    gf2_mult_233(b16, t1, b8);

    // b32 = (b16)^(2^16) * b16
    gf2_square_k(t1, b16, 16);
    gf2_mult_233(b32, t1, b16);

    // b64 = (b32)^(2^32) * b32
    gf2_square_k(t1, b32, 32);
    gf2_mult_233(b64, t1, b32);

    // b128 = (b64)^(2^64) * b64
    gf2_square_k(t1, b64, 64);
    gf2_mult_233(b128, t1, b64);

    // b192 = (b128)^(2^64) * b64
    gf2_square_k(t1, b128, 64);
    gf2_mult_233(b192, t1, b64);

    // b224 = (b192)^(2^32) * b32
    gf2_square_k(t1, b192, 32);
    gf2_mult_233(b224, t1, b32);

    // b232 = (b224)^(2^8) * b8
    gf2_square_k(t1, b224, 8);
    gf2_mult_233(b232, t1, b8);

    // inv = (b232)^2
    gf2_square(r, b232);
}


// ============================================================
// Scalar multiplication (Lopez-Dahab double & add)
// ============================================================
extern "C" __global__ void scalar_mult_233(
    const uint64_t *Px, const uint64_t *Py, const uint64_t *Pz,
    const uint64_t *a_curve, const uint64_t *b_curve,
    const uint64_t *k,
    uint64_t *Xout, uint64_t *Yout, uint64_t *Zout)
{
    uint64_t X[4] = {Px[0], Px[1], Px[2], Px[3]};
    uint64_t Y[4] = {Py[0], Py[1], Py[2], Py[3]};
    uint64_t Z[4] = {Pz[0], Pz[1], Pz[2], Pz[3]};
    uint64_t X2[4], Y2[4], Z2[4];

    // find the highest set bit
    int msb = -1;
    for (int i = 233 - 1; i >= 0; --i) {
        if (get_bit(k, i)) { msb = i; break; }
    }
    if (msb < 0) return;

    // main loop (skip first MSB bit)
    for (int i = msb - 1; i >= 0; --i) {
        point_double_233(X2, Y2, Z2, X, Y, Z, a_curve, b_curve);

        if (get_bit(k, i)) {
            point_add_233(X2, Y2, Z2, X2, Y2, Z2, Px, Py, a_curve);
        }

        for (int j = 0; j < 4; j++) {
            X[j] = X2[j];
            Y[j] = Y2[j];
            Z[j] = Z2[j];
        }
    }

    // projective -> affine
    for (int j = 0; j < 4; j++) {
        Xout[j] = X[j]; Yout[j] = Y[j]; Zout[j] = Z[j];
    }

    uint64_t Zinv[4], Zinv2[4], x_aff[4], y_aff[4];
    gf2_inv_itoh_tsujii_233(Zinv, Zout);
    gf2_square(Zinv2, Zinv);
    gf2_mult_233(x_aff, Xout, Zinv);
    gf2_mult_233(y_aff, Yout, Zinv2);

    for (int j = 0; j < 4; j++) {
        Xout[j] = x_aff[j];
        Yout[j] = y_aff[j];
        Zout[j] = 0;
    }
}

} // extern "C"
'''

# ============================================================
# Compile kernel
# ============================================================
print("=" * 70)
print("Full GPU Benchmark: GF(2^233) Scalar Multiplication")
print("Platform: NVIDIA Jetson Orin Nano Super")
print("=" * 70)

cuda_root = "/usr/local/cuda-12.6"
include_paths = [f"{cuda_root}/include", f"{cuda_root}/include/crt"]
options = ('--std=c++11',) + tuple(f'-I{p}' for p in include_paths)
mod = cp.RawModule(code=kernel_code, options=options)
scalar_kernel = mod.get_function("scalar_mult_233")

print("[OK] Kernel compiled successfully")

# ============================================================
# Get kernel resource info
# ============================================================
kernel_attrs = scalar_kernel.attributes
num_regs = kernel_attrs.get('num_regs', 'N/A')
shared_size_bytes = kernel_attrs.get('shared_size_bytes', 0)
const_size_bytes = kernel_attrs.get('const_size_bytes', 0)
local_size_bytes = kernel_attrs.get('local_size_bytes', 0)
max_threads_per_block = kernel_attrs.get('max_threads_per_block', 'N/A')
ptx_version = kernel_attrs.get('ptx_version', 'N/A')
binary_version = kernel_attrs.get('binary_version', 'N/A')

print(f"[OK] Kernel attributes retrieved")
print(f"     Registers/thread:       {num_regs}")
print(f"     Shared memory:          {shared_size_bytes} bytes")
print(f"     Constant memory:        {const_size_bytes} bytes")
print(f"     Local memory:           {local_size_bytes} bytes")
print(f"     Max threads/block:      {max_threads_per_block}")

# ============================================================
# GPU device info
# ============================================================
dev = cp.cuda.Device(0)
dev_props = cp.cuda.runtime.getDeviceProperties(0)
gpu_name = dev_props.get('name', b'Unknown').decode() if isinstance(dev_props.get('name', b''), bytes) else str(dev_props.get('name', 'Unknown'))
total_mem = dev_props.get('totalGlobalMem', 0)
sm_count = dev_props.get('multiProcessorCount', 0)
max_threads_per_sm = dev_props.get('maxThreadsPerMultiProcessor', 0)
clock_rate_khz = dev_props.get('clockRate', 0)
mem_clock_khz = dev_props.get('memoryClockRate', 0)
l2_cache = dev_props.get('l2CacheSize', 0)
warp_size = dev_props.get('warpSize', 32)
compute_major = dev_props.get('major', 0)
compute_minor = dev_props.get('minor', 0)
shared_mem_per_block = dev_props.get('sharedMemPerBlock', 0)
shared_mem_per_sm = dev_props.get('sharedMemPerMultiprocessor', 0)
regs_per_block = dev_props.get('regsPerBlock', 0)
regs_per_sm = dev_props.get('regsPerMultiprocessor', 0)
max_blocks_per_sm = dev_props.get('maxBlocksPerMultiProcessor', 0)

print(f"\n[GPU] {gpu_name}")
print(f"      Compute Capability:    {compute_major}.{compute_minor}")
print(f"      SMs:                   {sm_count}")
print(f"      GPU Clock:             {clock_rate_khz/1000:.0f} MHz")
print(f"      Memory Clock:          {mem_clock_khz/1000:.0f} MHz")
print(f"      Total Memory:          {total_mem / (1024**3):.2f} GB")
print(f"      L2 Cache:              {l2_cache / 1024:.0f} KB")

# ============================================================
# Setup GPU buffers
# ============================================================
Px = cp.asarray(Gx)
Py = cp.asarray(Gy)
Pz = cp.asarray(Gz)
a_gpu = cp.asarray(a_curve)
b_gpu = cp.asarray(b_curve)
Xout = cp.zeros(4, dtype=cp.uint64)
Yout = cp.zeros(4, dtype=cp.uint64)
Zout = cp.zeros(4, dtype=cp.uint64)

# ============================================================
# tegrastats monitor (background thread)
# ============================================================
tegra_samples = []
tegra_running = False

def tegrastats_monitor():
    global tegra_running
    proc = subprocess.Popen(
        ['tegrastats', '--interval', '200'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    while tegra_running:
        line = proc.stdout.readline()
        if line:
            tegra_samples.append(line.strip())
    proc.terminate()
    proc.wait()

def parse_tegrastats(samples):
    """Parse tegrastats output lines into structured data."""
    gpu_freqs = []
    gpu_temps = []
    cpu_temps = []
    vdd_in = []
    vdd_cpu_gpu = []
    vdd_soc = []
    ram_used = []
    cpu_loads = []

    for line in samples:
        # GR3D_FREQ
        m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
        if m:
            gpu_freqs.append(int(m.group(1)))

        # GPU temp
        m = re.search(r'gpu@([\d.]+)C', line)
        if m:
            gpu_temps.append(float(m.group(1)))

        # CPU temp
        m = re.search(r'cpu@([\d.]+)C', line)
        if m:
            cpu_temps.append(float(m.group(1)))

        # VDD_IN power
        m = re.search(r'VDD_IN\s+(\d+)mW', line)
        if m:
            vdd_in.append(int(m.group(1)))

        # VDD_CPU_GPU_CV power
        m = re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW', line)
        if m:
            vdd_cpu_gpu.append(int(m.group(1)))

        # VDD_SOC power
        m = re.search(r'VDD_SOC\s+(\d+)mW', line)
        if m:
            vdd_soc.append(int(m.group(1)))

        # RAM
        m = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
        if m:
            ram_used.append(int(m.group(1)))

        # CPU loads
        m = re.search(r'CPU\s+\[([^\]]+)\]', line)
        if m:
            cores = re.findall(r'(\d+)%', m.group(1))
            if cores:
                cpu_loads.append(np.mean([int(c) for c in cores]))

    return {
        'gpu_freq': gpu_freqs,
        'gpu_temp': gpu_temps,
        'cpu_temp': cpu_temps,
        'vdd_in': vdd_in,
        'vdd_cpu_gpu': vdd_cpu_gpu,
        'vdd_soc': vdd_soc,
        'ram_used': ram_used,
        'cpu_load': cpu_loads,
    }

# ============================================================
# Read power sensors directly
# ============================================================
def read_power_mw():
    """Read power from INA3221 sensor."""
    try:
        with open('/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input') as f:
            vdd_in_ma = int(f.read().strip())
        with open('/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input') as f:
            vdd_in_mv = int(f.read().strip())
        power_mw = vdd_in_ma * vdd_in_mv / 1000
        return {'vdd_in_mw': power_mw, 'vdd_in_ma': vdd_in_ma, 'vdd_in_mv': vdd_in_mv}
    except Exception:
        return None

# ============================================================
# Memory usage tracking
# ============================================================
def get_gpu_memory():
    mempool = cp.get_default_memory_pool()
    pinned_pool = cp.get_default_pinned_memory_pool()
    return {
        'used_bytes': mempool.used_bytes(),
        'total_bytes': mempool.total_bytes(),
        'pinned_bytes': pinned_pool.n_free_blocks(),
    }

# ============================================================
# WARMUP (5 runs, not measured)
# ============================================================
print("\n--- Warmup (5 runs) ---")
for w in range(5):
    k_int = random.getrandbits(M - 1) | (1 << (M - 1))
    k_limbs = [(k_int >> (64*i)) & ((1 << 64) - 1) for i in range(4)]
    k_gpu = cp.asarray(k_limbs, dtype=cp.uint64)
    scalar_kernel((1,), (1,), (Px, Py, Pz, a_gpu, b_gpu, k_gpu, Xout, Yout, Zout))
    cp.cuda.Device().synchronize()
    print(f"  Warmup {w+1}/5 done")

# ============================================================
# BENCHMARK: Latency measurement (50 runs)
# ============================================================
NUM_RUNS = 50
print(f"\n--- Benchmark ({NUM_RUNS} runs) ---")

# Capture idle power before benchmark
idle_power = read_power_mw()
time.sleep(0.5)

# Start tegrastats monitoring
tegra_running = True
tegra_thread = threading.Thread(target=tegrastats_monitor, daemon=True)
tegra_thread.start()
time.sleep(0.5)  # let tegrastats start

# Clear samples from startup
tegra_samples.clear()

wall_times = []
cuda_times = []
results_log = []
power_samples = []
mem_before = get_gpu_memory()

for t in range(NUM_RUNS):
    k_int = random.getrandbits(M - 1) | (1 << (M - 1))
    k_limbs = [(k_int >> (64*i)) & ((1 << 64) - 1) for i in range(4)]
    k_gpu = cp.asarray(k_limbs, dtype=cp.uint64)

    # CUDA events for precise GPU timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    # Wall clock
    wall_start = time.perf_counter()

    # GPU timing
    start_event.record()
    scalar_kernel((1,), (1,), (Px, Py, Pz, a_gpu, b_gpu, k_gpu, Xout, Yout, Zout))
    end_event.record()
    end_event.synchronize()

    wall_time = (time.perf_counter() - wall_start) * 1000
    cuda_time = cp.cuda.get_elapsed_time(start_event, end_event)

    wall_times.append(wall_time)
    cuda_times.append(cuda_time)

    # Read power during run
    pwr = read_power_mw()
    if pwr:
        power_samples.append(pwr['vdd_in_mw'])

    Xr = (int(Xout[3].get()) << 192) | (int(Xout[2].get()) << 128) | (int(Xout[1].get()) << 64) | int(Xout[0].get())
    results_log.append((t + 1, k_int, Xr, wall_time, cuda_time))

    if (t + 1) % 10 == 0:
        print(f"  Run {t+1}/{NUM_RUNS}: wall={wall_time:.2f}ms, cuda={cuda_time:.2f}ms")

mem_after = get_gpu_memory()

# Stop tegrastats
time.sleep(0.5)
tegra_running = False
tegra_thread.join(timeout=3)

# ============================================================
# Parse tegrastats data
# ============================================================
tegra_data = parse_tegrastats(tegra_samples)

# ============================================================
# Compute statistics
# ============================================================
def stats(data, name=""):
    arr = np.array(data, dtype=np.float64)
    s = sorted(arr)
    n = len(arr)
    return {
        'name': name,
        'n': n,
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'p5': np.percentile(arr, 5),
        'p25': np.percentile(arr, 25),
        'p75': np.percentile(arr, 75),
        'p95': np.percentile(arr, 95),
        'p99': np.percentile(arr, 99),
        'iqr': np.percentile(arr, 75) - np.percentile(arr, 25),
        'total': np.sum(arr),
    }

wall_stats = stats(wall_times, "Wall-clock latency")
cuda_stats = stats(cuda_times, "CUDA event latency")

# ============================================================
# Write full benchmark report
# ============================================================
report_path = os.path.join(os.getcwd(), "full_benchmark_233.txt")

with open(report_path, "w") as f:
    f.write("=" * 75 + "\n")
    f.write("FULL GPU BENCHMARK REPORT\n")
    f.write("GF(2^233) Scalar Multiplication - sect233r1\n")
    f.write("Lopez-Dahab Projective Coordinates + Itoh-Tsujii Inversion\n")
    f.write("=" * 75 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Benchmark Runs: {NUM_RUNS} (+ 5 warmup)\n\n")

    # --- Hardware Info ---
    f.write("-" * 75 + "\n")
    f.write("1. HARDWARE PLATFORM\n")
    f.write("-" * 75 + "\n")
    f.write(f"  GPU:                       {gpu_name}\n")
    f.write(f"  Compute Capability:        {compute_major}.{compute_minor}\n")
    f.write(f"  Streaming Multiprocessors: {sm_count}\n")
    f.write(f"  GPU Clock Rate:            {clock_rate_khz/1000:.0f} MHz\n")
    f.write(f"  Memory Clock Rate:         {mem_clock_khz/1000:.0f} MHz\n")
    f.write(f"  Total Global Memory:       {total_mem / (1024**3):.2f} GB\n")
    f.write(f"  L2 Cache Size:             {l2_cache / 1024:.0f} KB\n")
    f.write(f"  Warp Size:                 {warp_size}\n")
    f.write(f"  Max Threads/Block:         {dev_props.get('maxThreadsPerBlock', 'N/A')}\n")
    f.write(f"  Max Threads/SM:            {max_threads_per_sm}\n")
    f.write(f"  Max Blocks/SM:             {max_blocks_per_sm}\n")
    f.write(f"  Registers/Block:           {regs_per_block}\n")
    f.write(f"  Registers/SM:              {regs_per_sm}\n")
    f.write(f"  Shared Memory/Block:       {shared_mem_per_block / 1024:.1f} KB\n")
    f.write(f"  Shared Memory/SM:          {shared_mem_per_sm / 1024:.1f} KB\n")
    f.write(f"  CUDA Version:              {cp.cuda.runtime.runtimeGetVersion()}\n")
    f.write(f"  CuPy Version:              {cp.__version__}\n\n")

    # --- Kernel Resources ---
    f.write("-" * 75 + "\n")
    f.write("2. KERNEL RESOURCE USAGE\n")
    f.write("-" * 75 + "\n")
    f.write(f"  Kernel Name:               scalar_mult_233\n")
    f.write(f"  Grid Size:                 (1, 1, 1)\n")
    f.write(f"  Block Size:                (1, 1, 1)\n")
    f.write(f"  Registers/Thread:          {num_regs}\n")
    f.write(f"  Shared Memory (static):    {shared_size_bytes} bytes\n")
    f.write(f"  Constant Memory:           {const_size_bytes} bytes\n")
    f.write(f"  Local Memory/Thread:       {local_size_bytes} bytes\n")
    f.write(f"  Max Threads/Block (kernel):{max_threads_per_block}\n")
    f.write(f"  PTX Version:               {ptx_version}\n")
    f.write(f"  Binary Version:            {binary_version}\n")

    # Occupancy calculation
    if num_regs != 'N/A' and regs_per_sm > 0:
        # 1 thread, 1 block - occupancy is 1 warp / max warps per SM
        max_warps_per_sm = max_threads_per_sm // warp_size if warp_size else 0
        active_warps = 1  # we launch 1 thread = 1 warp
        occupancy = (active_warps / max_warps_per_sm * 100) if max_warps_per_sm else 0
        f.write(f"  Theoretical Occupancy:     {occupancy:.2f}% ({active_warps}/{max_warps_per_sm} warps)\n")
        f.write(f"  Note: Single-thread kernel - occupancy is intentionally low.\n")
        f.write(f"        This kernel performs a single ECC scalar multiplication.\n")

    f.write("\n")

    # --- GPU Memory Usage ---
    f.write("-" * 75 + "\n")
    f.write("3. GPU MEMORY USAGE\n")
    f.write("-" * 75 + "\n")
    f.write(f"  CuPy Pool - Used:          {mem_after['used_bytes'] / 1024:.2f} KB\n")
    f.write(f"  CuPy Pool - Total Alloc:   {mem_after['total_bytes'] / 1024:.2f} KB\n")
    f.write(f"  Input Buffers:             9 x 4 x uint64 = {9 * 4 * 8} bytes (Px,Py,Pz,a,b,k,Xout,Yout,Zout)\n")
    if tegra_data['ram_used']:
        f.write(f"  System RAM Used (avg):     {np.mean(tegra_data['ram_used']):.0f} MB\n")
    f.write("\n")

    # --- Latency ---
    f.write("-" * 75 + "\n")
    f.write("4. LATENCY MEASUREMENTS\n")
    f.write("-" * 75 + "\n\n")

    for s_data in [wall_stats, cuda_stats]:
        f.write(f"  {s_data['name']} ({s_data['n']} runs):\n")
        f.write(f"    Mean:                    {s_data['mean']:.4f} ms\n")
        f.write(f"    Median:                  {s_data['median']:.4f} ms\n")
        f.write(f"    Std Deviation:           {s_data['std']:.4f} ms\n")
        f.write(f"    Min:                     {s_data['min']:.4f} ms\n")
        f.write(f"    Max:                     {s_data['max']:.4f} ms\n")
        f.write(f"    5th Percentile:          {s_data['p5']:.4f} ms\n")
        f.write(f"    25th Percentile:         {s_data['p25']:.4f} ms\n")
        f.write(f"    75th Percentile:         {s_data['p75']:.4f} ms\n")
        f.write(f"    95th Percentile:         {s_data['p95']:.4f} ms\n")
        f.write(f"    99th Percentile:         {s_data['p99']:.4f} ms\n")
        f.write(f"    IQR:                     {s_data['iqr']:.4f} ms\n")
        f.write(f"    Total:                   {s_data['total']:.4f} ms\n")
        f.write(f"    Throughput:              {1000.0 / s_data['mean']:.2f} ops/sec\n\n")

    # Per-run table
    f.write("  Per-Run Latency Detail:\n")
    f.write(f"  {'Run':<6}{'Wall (ms)':<14}{'CUDA (ms)':<14}{'Scalar k (hex)':<63}{'x_aff (hex)'}\n")
    f.write("  " + "-" * 91 + "\n")
    for run, k_val, x_val, w_ms, c_ms in results_log:
        f.write(f"  {run:<6}{w_ms:<14.4f}{c_ms:<14.4f}0x{k_val:059x}   0x{x_val:059x}\n")
    f.write("\n")

    # --- Power Consumption ---
    f.write("-" * 75 + "\n")
    f.write("5. POWER CONSUMPTION\n")
    f.write("-" * 75 + "\n")

    if idle_power:
        f.write(f"  Idle Power (VDD_IN):       {idle_power['vdd_in_mw']:.0f} mW\n")

    if tegra_data['vdd_in']:
        vdd_arr = np.array(tegra_data['vdd_in'])
        f.write(f"\n  Under Load (tegrastats, {len(vdd_arr)} samples):\n")
        f.write(f"    VDD_IN (Total Board):\n")
        f.write(f"      Mean:                  {np.mean(vdd_arr):.0f} mW\n")
        f.write(f"      Min:                   {np.min(vdd_arr):.0f} mW\n")
        f.write(f"      Max:                   {np.max(vdd_arr):.0f} mW\n")
        f.write(f"      Std Dev:               {np.std(vdd_arr):.0f} mW\n")

    if tegra_data['vdd_cpu_gpu']:
        vdd_cg = np.array(tegra_data['vdd_cpu_gpu'])
        f.write(f"\n    VDD_CPU_GPU_CV:\n")
        f.write(f"      Mean:                  {np.mean(vdd_cg):.0f} mW\n")
        f.write(f"      Min:                   {np.min(vdd_cg):.0f} mW\n")
        f.write(f"      Max:                   {np.max(vdd_cg):.0f} mW\n")

    if tegra_data['vdd_soc']:
        vdd_s = np.array(tegra_data['vdd_soc'])
        f.write(f"\n    VDD_SOC:\n")
        f.write(f"      Mean:                  {np.mean(vdd_s):.0f} mW\n")
        f.write(f"      Min:                   {np.min(vdd_s):.0f} mW\n")
        f.write(f"      Max:                   {np.max(vdd_s):.0f} mW\n")

    if power_samples:
        ps = np.array(power_samples)
        f.write(f"\n  INA3221 Sensor Readings ({len(ps)} samples):\n")
        f.write(f"    Mean Power:              {np.mean(ps):.0f} mW\n")
        f.write(f"    Min Power:               {np.min(ps):.0f} mW\n")
        f.write(f"    Max Power:               {np.max(ps):.0f} mW\n")

    # Energy per operation
    if tegra_data['vdd_in'] and cuda_stats['mean'] > 0:
        avg_power_w = np.mean(tegra_data['vdd_in']) / 1000.0
        avg_time_s = cuda_stats['mean'] / 1000.0
        energy_mj = avg_power_w * avg_time_s * 1000
        energy_uj = energy_mj * 1000
        f.write(f"\n  Energy per Scalar Multiplication:\n")
        f.write(f"    Avg Power:               {avg_power_w*1000:.0f} mW\n")
        f.write(f"    Avg CUDA Time:           {cuda_stats['mean']:.4f} ms\n")
        f.write(f"    Energy/Op:               {energy_mj:.4f} mJ ({energy_uj:.2f} uJ)\n")

    f.write("\n")

    # --- Thermal ---
    f.write("-" * 75 + "\n")
    f.write("6. THERMAL MONITORING\n")
    f.write("-" * 75 + "\n")

    if tegra_data['gpu_temp']:
        gt = np.array(tegra_data['gpu_temp'])
        f.write(f"  GPU Temperature ({len(gt)} samples):\n")
        f.write(f"    Mean:                    {np.mean(gt):.1f} C\n")
        f.write(f"    Min:                     {np.min(gt):.1f} C\n")
        f.write(f"    Max:                     {np.max(gt):.1f} C\n")

    if tegra_data['cpu_temp']:
        ct = np.array(tegra_data['cpu_temp'])
        f.write(f"\n  CPU Temperature ({len(ct)} samples):\n")
        f.write(f"    Mean:                    {np.mean(ct):.1f} C\n")
        f.write(f"    Min:                     {np.min(ct):.1f} C\n")
        f.write(f"    Max:                     {np.max(ct):.1f} C\n")

    f.write("\n")

    # --- GPU Utilization ---
    f.write("-" * 75 + "\n")
    f.write("7. GPU UTILIZATION\n")
    f.write("-" * 75 + "\n")

    if tegra_data['gpu_freq']:
        gf = np.array(tegra_data['gpu_freq'])
        f.write(f"  GR3D GPU Frequency Load ({len(gf)} samples):\n")
        f.write(f"    Mean:                    {np.mean(gf):.1f}%\n")
        f.write(f"    Min:                     {np.min(gf):.0f}%\n")
        f.write(f"    Max:                     {np.max(gf):.0f}%\n")

    if tegra_data['cpu_load']:
        cl = np.array(tegra_data['cpu_load'])
        f.write(f"\n  CPU Average Load ({len(cl)} samples):\n")
        f.write(f"    Mean:                    {np.mean(cl):.1f}%\n")
        f.write(f"    Min:                     {np.min(cl):.1f}%\n")
        f.write(f"    Max:                     {np.max(cl):.1f}%\n")

    f.write("\n")

    # --- Summary ---
    f.write("=" * 75 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 75 + "\n")
    f.write(f"  Curve:                     sect233r1 (GF(2^233))\n")
    f.write(f"  Algorithm:                 Lopez-Dahab double-and-add\n")
    f.write(f"  Inversion:                 Itoh-Tsujii (addition chain)\n")
    f.write(f"  Field Representation:      4 x 64-bit limbs (polynomial basis)\n")
    f.write(f"  GPU:                       {gpu_name}\n")
    f.write(f"  Avg Latency (CUDA):        {cuda_stats['mean']:.4f} ms\n")
    f.write(f"  Avg Latency (Wall):        {wall_stats['mean']:.4f} ms\n")
    f.write(f"  Throughput:                {1000.0 / cuda_stats['mean']:.2f} ops/sec\n")
    f.write(f"  Registers/Thread:          {num_regs}\n")
    if tegra_data['vdd_in']:
        f.write(f"  Avg Board Power:           {np.mean(tegra_data['vdd_in']):.0f} mW\n")
    if tegra_data['gpu_temp']:
        f.write(f"  Avg GPU Temp:              {np.mean(tegra_data['gpu_temp']):.1f} C\n")
    if tegra_data['vdd_in'] and cuda_stats['mean'] > 0:
        f.write(f"  Energy/Op:                 {energy_mj:.4f} mJ\n")
    f.write("=" * 75 + "\n")

print(f"\n[SAVED] Full benchmark report: {report_path}")
print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
