# -*- coding: utf-8 -*-
"""
GPU Lopez–Dahab Scalar Multiplication over GF(2^571)
Curve: sect571r1  (y^2 + xy = x^3 + ax^2 + b)
f(x) = x^571 + x^10 + x^5 + x^2 + 1
Author: Salah Harb
"""

import cupy as cp
import numpy as np
import time, random

# ============================================================
# Parameters for sect571r1
# ============================================================
M = 571
IRRED_POLY = (1 << 571) | (1 << 10) | (1 << 5) | (1 << 2) | 1

a_curve_int = 1
b_curve_int = 0x02f40e7e2221f295de297117b7f3d62f5c6a97ffcb8ceff1cd6ba8ce4a9a18ad84ffabbd8efa59332be7ad6756a66e294afd185a78ff12aa520e4de739baca0c7ffeff7f2955727a
Gx_int = 0x303001d34b856296c16c0d40d3cd7750a93d1d2955fa80aa5f40fc8db7b2abdbde53950f4c0d293cdd711a35b67fb1499ae60038614f1394abfa3b4c850d927e1e7769c8eec2d19
Gy_int = 0x37bf27342da639b6dccfffeb73d69d78c6c27a6009cbbca1980f8533921e8a684423e43bab08a576291af8f461bb2a8b3531d2f0485c19b16e2f1516e23dd3c1a4827af1b8ac15b
Gz_int = 1


# ============================================================
# 9-limb (576-bit) packing
# ============================================================
def to_limbs(x):
    return np.array([
        np.uint64((x >> (64 * i)) & ((1 << 64) - 1)) for i in range(9)
    ], dtype=np.uint64)



Gx = to_limbs(Gx_int)
Gy = to_limbs(Gy_int)
Gz = to_limbs(Gz_int)
a_curve = to_limbs(a_curve_int)
b_curve = to_limbs(b_curve_int)

print("===================================================")
print("Base Point P (sect571r1):")
print(f"Gx = 0x{Gx_int:059x}")
print(f"Gy = 0x{Gy_int:059x}")
print(f"Gz = 0x{Gz_int:059x}")
print("===================================================")

# ============================================================
# GPU kernel code (corrected 4-limb consistency)
# ============================================================
kernel_code = r'''
#ifndef __UINT64_T_DEFINED
typedef unsigned long long uint64_t;
#define __UINT64_T_DEFINED
#endif


extern "C" {

// ============================================================
// Basic helpers
// ============================================================
__device__ void print_hex(const char* name, const uint64_t *v) {
    printf("%s = 0x%016llx%016llx%016llx%016llx\n", name, v[3], v[2], v[1], v[0]);
}

    
// ============================================================
// GF(2) 64-bit multiply
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
// XOR combine helper
// ============================================================
__device__ void xor_shift(uint64_t *t, const uint64_t *val,
                          int nval, int shift) {
    int w = shift / 64;
    int b = shift % 64;
    for (int i = 0; i < nval; i++) {
        t[w + i] ^= val[i] << b;
        if (b && (w + i + 1 < 18))
            t[w + i + 1] ^= val[i] >> (64 - b);
    }
}

// ============================================================
// Multiply + reduction mod x^571 + x^10 + x^5 + x^2 + 1
// ============================================================
__device__ void gf2_mult_571(uint64_t *r, const uint64_t *a,
                             const uint64_t *b) {
    uint64_t t[18] = {0};

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            uint64_t hi, lo;
            gf2_mul64(&hi, &lo, a[i], b[j]);
            xor_shift(t, &lo, 1, (i + j) * 64);
            xor_shift(t, &hi, 1, (i + j) * 64 + 64);
        }
    }

    // Reduction mod x^571 + x^10 + x^5 + x^2 + 1
    for (int bit = 1151; bit >= 571; --bit) {
        int limb = bit / 64;
        int pos  = bit % 64;
        if (t[limb] & (1ULL << pos)) {
            t[limb] ^= (1ULL << pos);
            int shift = bit - 571;
            int targets[4] = {shift + 10, shift + 5, shift + 2, shift};
            for (int k = 0; k < 4; k++) {
                int tgt = targets[k];
                if (tgt < 0) continue;
                int w = tgt / 64, bpos = tgt % 64;
                t[w] ^= (1ULL << bpos);
            }
        }
    }

    for (int i = 0; i < 9; i++) r[i] = t[i];
    r[8] &= ((1ULL << 59) - 1); // top 571 bits only
}

// ============================================================
// Squaring
// ============================================================
__device__ void gf2_square(uint64_t *r, const uint64_t *a) {
    uint64_t t[18] = {0};

    for (int limb = 0; limb < 9; limb++) {
        uint64_t x = a[limb];
        for (int bit = 0; bit < 64; bit++) {
            if (x & (1ULL << bit)) {
                int pos = 2 * (bit + 64 * limb);
                int w = pos / 64;
                int bpos = pos % 64;
                t[w] ^= (1ULL << bpos);
            }
        }
    }

    for (int bit = 1151; bit >= 571; --bit) {
        int limb = bit / 64;
        int pos  = bit % 64;
        if (t[limb] & (1ULL << pos)) {
            t[limb] ^= (1ULL << pos);
            int shift = bit - 571;
            int targets[4] = {shift + 10, shift + 5, shift + 2, shift};
            for (int k = 0; k < 4; k++) {
                int tgt = targets[k];
                if (tgt < 0) continue;
                int w = tgt / 64, bpos = tgt % 64;
                t[w] ^= (1ULL << bpos);
            }
        }
    }

    for (int i = 0; i < 9; i++) r[i] = t[i];
    r[8] &= ((1ULL << 59) - 1);
}

// ============================================================
// k-fold squaring
// ============================================================
__device__ void gf2_square_k(uint64_t *r, const uint64_t *a, int k) {
    uint64_t tmp1[9], tmp2[9];
    for (int i = 0; i < 9; i++) tmp1[i] = a[i];
    for (int i = 0; i < k; i++) {
        gf2_square(tmp2, tmp1);
        for (int j = 0; j < 9; j++) tmp1[j] = tmp2[j];
    }
    for (int j = 0; j < 9; j++) r[j] = tmp1[j];
}

__device__ bool get_bit(const uint64_t *k, int i) {
    int limb = i / 64;
    int bit  = i % 64;
    return (k[limb] >> bit) & 1ULL;
}


// ============================================================
// Itoh–Tsujii inversion µ-sequence for GF(2^571)
// ============================================================
__device__ void gf2_inv_itoh_tsujii_571(uint64_t *r, const uint64_t *a) {
    uint64_t t1[9], t2[9];
    uint64_t b1[9]; for (int i = 0; i < 9; i++) b1[i] = a[i];
    uint64_t b2[9], b4[9], b8[9], b16[9], b32[9], b64[9], b128[9], b256[9], b512[9];
    uint64_t tmp[9];

    // Core chain
    gf2_square(t1, b1);  gf2_mult_571(b2, t1, b1);
    gf2_square_k(t1, b2, 2);  gf2_mult_571(b4, t1, b2);
    gf2_square_k(t1, b4, 4);  gf2_mult_571(b8, t1, b4);
    gf2_square_k(t1, b8, 8);  gf2_mult_571(b16, t1, b8);
    gf2_square_k(t1, b16, 16); gf2_mult_571(b32, t1, b16);
    gf2_square_k(t1, b32, 32); gf2_mult_571(b64, t1, b32);
    gf2_square_k(t1, b64, 64); gf2_mult_571(b128, t1, b64);
    gf2_square_k(t1, b128, 128); gf2_mult_571(b256, t1, b128);
    gf2_square_k(t1, b256, 256); gf2_mult_571(b512, t1, b256);

    // Extend 512 → 570
    gf2_square_k(tmp, b512, 32); gf2_mult_571(tmp, tmp, b32);
    gf2_square_k(tmp, tmp, 16); gf2_mult_571(tmp, tmp, b16);
    gf2_square_k(tmp, tmp, 8);  gf2_mult_571(tmp, tmp, b8);
    gf2_square_k(tmp, tmp, 2);  gf2_mult_571(tmp, tmp, b2);

    // Final square
    gf2_square(r, tmp);
}



// ============================================================
// Lopez–Dahab Point Double for GF(2^571)
// ============================================================ 
__device__ void point_double_571(
    uint64_t *X2, uint64_t *Y2, uint64_t *Z2,
    const uint64_t *X1, const uint64_t *Y1, const uint64_t *Z1,
    const uint64_t *a_curve, const uint64_t *b_curve)
{
    uint64_t X1_sq[9], Z1_sq[9], Z1_4[9], X1_4[9], Y1_sq[9];
    uint64_t bZ1_4[9], aZ2[9], inner[9], right[9];
    uint64_t X2right[9], left[9];

    printf("\n=== Point Doubling (GF(2^571)) ===\n");
    print_hex("Input X1", X1);
    print_hex("Input Y1", Y1);
    print_hex("Input Z1", Z1);
    print_hex("a_curve", a_curve);
    print_hex("b_curve", b_curve);
    printf("----------------------------------\n");

    // Step 1–5: Squares
    gf2_square(X1_sq, X1);
    gf2_square(Z1_sq, Z1);
    gf2_square(Z1_4, Z1_sq);
    gf2_square(X1_4, X1_sq);
    gf2_square(Y1_sq, Y1);
    print_hex("Y1_sq", Y1_sq);

    // Step 6: Z2 = X1^2 * Z1^2
    gf2_mult_571(Z2, X1_sq, Z1_sq);
    print_hex("Z2", Z2);

    // Step 7: bZ1_4 = b * Z1^4
    gf2_mult_571(bZ1_4, b_curve, Z1_4);
    print_hex("bZ1_4", bZ1_4);

    // Step 8: X2 = X1^4 + bZ1_4
    for (int i = 0; i < 9; i++) X2[i] = X1_4[i] ^ bZ1_4[i];
    print_hex("X2", X2);

    // Step 9: left = (bZ1_4 * Z2)
    gf2_mult_571(left, bZ1_4, Z2);
    print_hex("left", left);

    // Step 10: aZ2 = a * Z2
    gf2_mult_571(aZ2, a_curve, Z2);
    print_hex("aZ2", aZ2);

    // Step 11: inner = aZ2 + Y1^2
    for (int i = 0; i < 9; i++) inner[i] = aZ2[i] ^ Y1_sq[i];
    print_hex("inner (aZ2+Y1_sq)", inner);

    // Step 12: right = inner + bZ1_4
    for (int i = 0; i < 9; i++) right[i] = inner[i] ^ bZ1_4[i];
    print_hex("right (=aZ2+Y1_sq+bZ1_4)", right);

    // Step 13: X2right = X2 * right
    gf2_mult_571(X2right, X2, right);
    print_hex("X2right", X2right);

    // Step 14: Y2 = left + X2right
    for (int i = 0; i < 9; i++) Y2[i] = left[i] ^ X2right[i];
    print_hex("Y2", Y2);

    printf("=== End Point Doubling ===\n");
}

// ============================================================
// Lopez–Dahab Point Add (Z1 = 1)  for GF(2^571)
// ============================================================
__device__ void point_add_571(
    uint64_t *X2, uint64_t *Y2, uint64_t *Z2,
    const uint64_t *X0, const uint64_t *Y0, const uint64_t *Z0,
    const uint64_t *X1, const uint64_t *Y1,
    const uint64_t *a_curve)
{
    uint64_t T1[9], T2[9], A[9], T3[9], B[9], C[9];
    uint64_t B2[9], T4[9], T5[9], D[9], Z2t[9], E[9], A2[9], T6[9], F[9], G[9];
    uint64_t T7[9], T8[9], T9[9], T10[9];

    printf("\n=== Point Addition (GF(2^571)) ===\n");
    print_hex("X0", X0);
    print_hex("Y0", Y0);
    print_hex("Z0", Z0);
    print_hex("X1", X1);
    print_hex("Y1", Y1);
    printf("----------------------------------\n");

    // Step 1: A = Y1 * Z0^2 + Y0
    gf2_square(T1, Z0);
    print_hex("T1 (Z0^2)", T1);

    gf2_mult_571(T2, Y1, T1);
    print_hex("T2 (Y1*Z0^2)", T2);

    for (int i = 0; i < 9; i++) A[i] = T2[i] ^ Y0[i];
    print_hex("A", A);

    // Step 2: B = X1 * Z0 + X0
    gf2_mult_571(T3, X1, Z0);
    print_hex("T3 (X1*Z0)", T3);

    for (int i = 0; i < 9; i++) B[i] = T3[i] ^ X0[i];
    print_hex("B", B);

    // Step 3: C = Z0 * B
    gf2_mult_571(C, Z0, B);
    print_hex("C (Z0*B)", C);

    // Step 4: D = B^2 * (C + a * Z0^2)
    gf2_square(B2, B);
    print_hex("B2 (B^2)", B2);

    gf2_mult_571(T4, a_curve, T1);
    print_hex("T4 (a*Z0^2)", T4);

    for (int i = 0; i < 9; i++) T5[i] = C[i] ^ T4[i];
    print_hex("T5 (C+a*Z0^2)", T5);

    gf2_mult_571(D, B2, T5);
    print_hex("D", D);

    // Step 5: Z2 = C^2
    gf2_square(Z2t, C);
    print_hex("Z2t (C^2)", Z2t);

    // Step 6: E = A * C
    gf2_mult_571(E, A, C);
    print_hex("E (A*C)", E);

    // Step 7: X2 = A^2 + D + E
    gf2_square(A2, A);
    print_hex("A2 (A^2)", A2);

    for (int i = 0; i < 9; i++) T6[i] = A2[i] ^ D[i];
    print_hex("T6 (A^2+D)", T6);

    for (int i = 0; i < 9; i++) X2[i] = T6[i] ^ E[i];
    print_hex("X2", X2);

    // Step 8: F = X2 + X1 * Z2
    gf2_mult_571(T7, X1, Z2t);
    print_hex("T7 (X1*Z2)", T7);

    for (int i = 0; i < 9; i++) F[i] = X2[i] ^ T7[i];
    print_hex("F", F);

    // Step 9: G = X2 + Y1 * Z2
    gf2_mult_571(T8, Y1, Z2t);
    print_hex("T8 (Y1*Z2)", T8);

    for (int i = 0; i < 9; i++) G[i] = X2[i] ^ T8[i];
    print_hex("G", G);

    // Step 10: Y2 = E*F + Z2*G
    gf2_mult_571(T9, E, F);
    print_hex("T9 (E*F)", T9);

    gf2_mult_571(T10, Z2t, G);
    print_hex("T10 (Z2*G)", T10);

    for (int i = 0; i < 9; i++) Y2[i] = T9[i] ^ T10[i];
    print_hex("Y2", Y2);

    // Step 11: Copy Z2t → Z2
    for (int i = 0; i < 9; i++) Z2[i] = Z2t[i];
    print_hex("Z2", Z2);

    printf("=== End Point Addition ===\n");
}

// ============================================================
// Scalar multiplication (Lopez–Dahab double & add)  GF(2^571)
// ============================================================
extern "C" __global__ void scalar_mult_571(
    const uint64_t *Px, const uint64_t *Py, const uint64_t *Pz,
    const uint64_t *a_curve, const uint64_t *b_curve,
    const uint64_t *k,
    uint64_t *Xout, uint64_t *Yout, uint64_t *Zout)
{
    uint64_t X[9] = {Px[0], Px[1], Px[2], Px[3], Px[4], Px[5], Px[6], Px[7], Px[8]};
    uint64_t Y[9] = {Py[0], Py[1], Py[2], Py[3], Py[4], Py[5], Py[6], Py[7], Py[8]};
    uint64_t Z[9] = {Pz[0], Pz[1], Pz[2], Pz[3], Pz[4], Pz[5], Pz[6], Pz[7], Pz[8]};
    uint64_t X2[9], Y2[9], Z2[9];

    // ===== find the highest set bit =====
    int msb = -1;
    for (int i = 571 - 1; i >= 0; --i) {
        if (get_bit(k, i)) { msb = i; break; }
    }
    if (msb < 0) { printf("k = 0\n"); return; }

    printf("\n=== Scalar Multiplication (GF(2^571)) ===\n");
    printf("MSB = %d\n", msb);
    printf("----------------------------------------\n");

    // ===== main loop (skip first MSB bit) =====
    for (int i = msb - 1; i >= 0; --i) {
        printf("[bit %d] Doubling\n", i);
        point_double_571(X2, Y2, Z2, X, Y, Z, a_curve, b_curve);

        if (get_bit(k, i)) {
            printf("[bit %d] Addition triggered\n", i);
            point_add_571(X2, Y2, Z2, X2, Y2, Z2, Px, Py, a_curve);
        }

        for (int j = 0; j < 9; j++) {
            X[j] = X2[j];
            Y[j] = Y2[j];
            Z[j] = Z2[j];
        }
    }

    // ===== projective → affine (unchanged) =====
    for (int j = 0; j < 9; j++) {
        Xout[j] = X[j];
        Yout[j] = Y[j];
        Zout[j] = Z[j];
    }

    uint64_t Zinv[9], Zinv2[9], x_aff[9], y_aff[9];
    gf2_inv_itoh_tsujii_571(Zinv, Zout);
    gf2_square(Zinv2, Zinv);
    gf2_mult_571(x_aff, Xout, Zinv);
    gf2_mult_571(y_aff, Yout, Zinv2);

    for (int j = 0; j < 9; j++) {
        Xout[j] = x_aff[j];
        Yout[j] = y_aff[j];
        Zout[j] = 0;
    }

    printf("\n=== Final Affine Point ===\n");
    print_hex("x_aff", x_aff);
    print_hex("y_aff", y_aff);
    printf("========================================\n");
}


} // extern "C"
'''

# ============================================================
# Compile and Run
# ============================================================
cuda_root = "/usr/local/cuda-12.6"
include_paths = [f"{cuda_root}/include", f"{cuda_root}/include/crt"]
options = ('--std=c++11',) + tuple(f'-I{p}' for p in include_paths)
mod = cp.RawModule(code=kernel_code, options=options)
scalar_kernel = mod.get_function("scalar_mult_571")

Px, Py, Pz = map(cp.asarray, (Gx, Gy, Gz))
a_gpu, b_gpu = map(cp.asarray, (a_curve, b_curve))
Xout, Yout, Zout = (cp.zeros(9, dtype=cp.uint64) for _ in range(3))

import matplotlib.pyplot as plt

# ============================================================
# Run multiple random scalar multiplications (GPU)
# ============================================================
num_tests = 50
times = []
results_log = []
hex_width = (M + 3) // 4

for t in range(num_tests):
    k_int = random.getrandbits(M - 1) | (1 << (M - 1))
    k_limbs = to_limbs(k_int)
    k_gpu = cp.asarray(k_limbs, dtype=cp.uint64)

    start = time.perf_counter()
    scalar_kernel((1,), (1,), (Px, Py, Pz, a_gpu, b_gpu, k_gpu, Xout, Yout, Zout))
    cp.cuda.Device().synchronize()
    gpu_time = (time.perf_counter() - start) * 1000
    times.append(gpu_time)

    Xr = sum(int(Xout[i].get()) << (64 * i) for i in range(9))
    Yr = sum(int(Yout[i].get()) << (64 * i) for i in range(9))

    results_log.append((t + 1, k_int, Xr, Yr, gpu_time))
    print(f"Test #{t+1:02d} | GPU time: {gpu_time:.2f} ms | x_aff = 0x{Xr:0{hex_width}x}")

# ============================================================
# Statistics
# ============================================================
avg_time = sum(times) / len(times)
min_time = min(times)
max_time = max(times)
min_idx = times.index(min_time) + 1
max_idx = times.index(max_time) + 1
median_time = sorted(times)[len(times) // 2]
total_time = sum(times)
std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

print("\n" + "=" * 60)
print("GPU Scalar Multiplication Performance - GF(2^571)")
print("=" * 60)
print(f"Number of runs : {num_tests}")
print(f"Average time   : {avg_time:.2f} ms")
print(f"Median time    : {median_time:.2f} ms")
print(f"Min time       : {min_time:.2f} ms (run #{min_idx})")
print(f"Max time       : {max_time:.2f} ms (run #{max_idx})")
print(f"Std deviation  : {std_dev:.2f} ms")
print(f"Total time     : {total_time:.2f} ms")
print("=" * 60)

# ============================================================
# Write statistics to txt file
# ============================================================
stats_file = "gpu_571_performance_stats.txt"
with open(stats_file, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("GPU Scalar Multiplication Performance - GF(2^571) sect571r1\n")
    f.write("Lopez-Dahab Projective Coordinates + Itoh-Tsujii Inversion\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Number of runs     : {num_tests}\n")
    f.write(f"Average time       : {avg_time:.4f} ms\n")
    f.write(f"Median time        : {median_time:.4f} ms\n")
    f.write(f"Min time           : {min_time:.4f} ms (run #{min_idx})\n")
    f.write(f"Max time           : {max_time:.4f} ms (run #{max_idx})\n")
    f.write(f"Std deviation      : {std_dev:.4f} ms\n")
    f.write(f"Total time         : {total_time:.4f} ms\n\n")

    f.write("-" * 70 + "\n")
    f.write(f"{'Run':<6}{'Time (ms)':<14}{'Scalar k (hex)':<148}{'x_aff (hex)'}\n")
    f.write("-" * 70 + "\n")
    for run, k_val, x_val, y_val, t_ms in results_log:
        f.write(f"{run:<6}{t_ms:<14.4f}0x{k_val:0{hex_width}x}   0x{x_val:0{hex_width}x}\n")

    f.write("-" * 70 + "\n")
    f.write(f"\nAll times (ms): {[round(t, 2) for t in times]}\n")

print(f"\n[SAVED] Statistics written to: {stats_file}")

# ============================================================
# Plot runs vs times
# ============================================================
plt.figure(figsize=(14, 6))
runs = list(range(1, num_tests + 1))

plt.plot(runs, times, marker='o', linestyle='-', linewidth=1.5, markersize=5, color='#2E86AB', label='GPU Time')
plt.axhline(y=avg_time, color='red', linestyle='--', linewidth=1.5, label=f'Average: {avg_time:.2f} ms')
plt.axhline(y=min_time, color='green', linestyle=':', linewidth=1, label=f'Min: {min_time:.2f} ms')
plt.axhline(y=max_time, color='orange', linestyle=':', linewidth=1, label=f'Max: {max_time:.2f} ms')

plt.fill_between(runs, avg_time - std_dev, avg_time + std_dev, alpha=0.15, color='red', label=f'Std Dev: {std_dev:.2f} ms')

plt.xlabel('Run Number', fontsize=12, fontweight='bold')
plt.ylabel('Time (ms)', fontsize=12, fontweight='bold')
plt.title('GPU Scalar Multiplication Performance\nGF(2^571) - sect571r1 - Lopez-Dahab (50 Runs)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper right')
plt.xticks(range(1, num_tests + 1, 2))
plt.tight_layout()

plot_file = 'gpu_571_performance_plot.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"[SAVED] Plot saved as: {plot_file}")
