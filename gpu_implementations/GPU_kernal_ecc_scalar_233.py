# -*- coding: utf-8 -*-
"""
GPU Lopez–Dahab Scalar Multiplication over GF(2^233)
Curve: sect233r1  (y^2 + xy = x^3 + ax^2 + b)
f(x) = x^233 + x^74 + 1
Author: Salah Harb
"""

import cupy as cp
import numpy as np
import time, random

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

# ============================================================
# Convert to 4×64-bit limbs
# ============================================================
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

print("===================================================")
print("Base Point P (sect233r1):")
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
// GF(2) bitwise multiply for ≤64-bit chunks
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
// XOR combine t[] ← t[] ⊕ (val << shift)
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
// Karatsuba 233-bit carryless multiply (4×4 limb)
// ============================================================
__device__ void gf2_mult_233(uint64_t *r, const uint64_t *a,
                             const uint64_t *b) {
    uint64_t t[8] = {0};

    // 4×4 schoolbook — safe baseline for correctness
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
// Lopez–Dahab Point Double
// ============================================================ 

__device__ void point_double_233(
    uint64_t *X2, uint64_t *Y2, uint64_t *Z2,
    const uint64_t *X1, const uint64_t *Y1, const uint64_t *Z1,
    const uint64_t *a_curve, const uint64_t *b_curve)
{
    uint64_t X1_sq[4], Z1_sq[4], Z1_4[4], X1_4[4], Y1_sq[4];
    uint64_t bZ1_4[4], aZ2[4], inner[4], right[4];
    uint64_t X2right[4], left[4];

    printf("\n=== Point Doubling (GF(2^233)) ===\n");
    print_hex("Input X1", X1);
    print_hex("Input Y1", Y1);
    print_hex("Input Z1", Z1);
    print_hex("a_curve", a_curve);
    print_hex("b_curve", b_curve);
    printf("----------------------------------\n");


    printf("\n=== Point Doubling (GF(2^233)) ===\n");

    // Step 1–5: squares
    gf2_square(X1_sq, X1);
    gf2_square(Z1_sq, Z1);
    gf2_square(Z1_4, Z1_sq);
    gf2_square(X1_4, X1_sq);
    gf2_square(Y1_sq, Y1);
    print_hex("Y1_sq", Y1_sq);

    // Step 6: Z2 = X1^2 * Z1^2
    gf2_mult_233(Z2, X1_sq, Z1_sq);
    print_hex("Z2", Z2);

    // Step 7: bZ1_4 = b * Z1^4
    gf2_mult_233(bZ1_4, b_curve, Z1_4);
    print_hex("bZ1_4", bZ1_4);

    // Step 8: X2 = X1^4 + bZ1_4
    for (int i = 0; i < 4; i++) X2[i] = X1_4[i] ^ bZ1_4[i];
    print_hex("X2", X2);

    // Step 9: left = (bZ1_4 * Z2)
    gf2_mult_233(left, bZ1_4, Z2);
    print_hex("left", left);

    // Step 10: aZ2 = a * Z2
    gf2_mult_233(aZ2, a_curve, Z2);
    print_hex("aZ2", aZ2);

    // Step 11: inner = aZ2 + Y1^2
    for (int i = 0; i < 4; i++) inner[i] = aZ2[i] ^ Y1_sq[i];
    print_hex("inner (aZ2+Y1_sq)", inner);

    // Step 12: right = inner + bZ1_4
    for (int i = 0; i < 4; i++) right[i] = inner[i] ^ bZ1_4[i];
    print_hex("right (=aZ2+Y1_sq+bZ1_4)", right);

    // Step 13: X2right = X2 * right
    gf2_mult_233(X2right, X2, right);
    print_hex("X2right", X2right);

    // Step 14: Y2 = left + X2right
    for (int i = 0; i < 4; i++) Y2[i] = left[i] ^ X2right[i];
    print_hex("Y2", Y2);

    printf("=== End Point Doubling ===\n");
}

// ============================================================
// Lopez–Dahab Point Add (Z1=1)
// ============================================================
__device__ void point_add_233(uint64_t *X2, uint64_t *Y2, uint64_t *Z2,
                              const uint64_t *X0, const uint64_t *Y0, const uint64_t *Z0,
                              const uint64_t *X1, const uint64_t *Y1,
                              const uint64_t *a_curve) {
    uint64_t T1[4], T2[4], A[4], T3[4], B[4], C[4];
    uint64_t B2[4], T4[4], T5[4], D[4], Z2t[4], E[4], A2[4], T6[4], F[4], G[4];
    uint64_t T7[4], T8[4], T9[4], T10[4];

    printf("\n=== Point Addition (GF(2^233)) ===\n");
    print_hex("X0", X0);
    print_hex("Y0", Y0);
    print_hex("Z0", Z0);
    print_hex("X1", X1);
    print_hex("Y1", Y1);
    printf("----------------------------------\n");

    // Step 1: A = Y1 * Z0^2 + Y0
    gf2_square(T1, Z0);
    print_hex("T1 (Z0^2)", T1);

    gf2_mult_233(T2, Y1, T1);
    print_hex("T2 (Y1*Z0^2)", T2);

    for (int i = 0; i < 4; i++) A[i] = T2[i] ^ Y0[i];
    print_hex("A", A);

    // Step 2: B = X1 * Z0 + X0
    gf2_mult_233(T3, X1, Z0);
    print_hex("T3 (X1*Z0)", T3);

    for (int i = 0; i < 4; i++) B[i] = T3[i] ^ X0[i];
    print_hex("B", B);

    // Step 3: C = Z0 * B
    gf2_mult_233(C, Z0, B);
    print_hex("C (Z0*B)", C);

    // Step 4: D = B^2 * (C + a * Z0^2)
    gf2_square(B2, B);
    print_hex("B2 (B^2)", B2);

    gf2_mult_233(T4, a_curve, T1);
    print_hex("T4 (a*Z0^2)", T4);

    for (int i = 0; i < 4; i++) T5[i] = C[i] ^ T4[i];
    print_hex("T5 (C+a*Z0^2)", T5);

    gf2_mult_233(D, B2, T5);
    print_hex("D", D);

    // Step 5: Z2 = C^2
    gf2_square(Z2t, C);
    print_hex("Z2t (C^2)", Z2t);

    // Step 6: E = A * C
    gf2_mult_233(E, A, C);
    print_hex("E (A*C)", E);

    // Step 7: X2 = A^2 + D + E
    gf2_square(A2, A);
    print_hex("A2 (A^2)", A2);

    for (int i = 0; i < 4; i++) T6[i] = A2[i] ^ D[i];
    print_hex("T6 (A^2+D)", T6);

    for (int i = 0; i < 4; i++) X2[i] = T6[i] ^ E[i];
    print_hex("X2", X2);

    // Step 8: F = X2 + X1 * Z2
    gf2_mult_233(T7, X1, Z2t);
    print_hex("T7 (X1*Z2)", T7);

    for (int i = 0; i < 4; i++) F[i] = X2[i] ^ T7[i];
    print_hex("F", F);

    // Step 9: G = X2 + Y1 * Z2
    gf2_mult_233(T8, Y1, Z2t);
    print_hex("T8 (Y1*Z2)", T8);

    for (int i = 0; i < 4; i++) G[i] = X2[i] ^ T8[i];
    print_hex("G", G);

    // Step 10: Y2 = E*F + Z2*G
    gf2_mult_233(T9, E, F);
    print_hex("T9 (E*F)", T9);

    gf2_mult_233(T10, Z2t, G);
    print_hex("T10 (Z2*G)", T10);

    for (int i = 0; i < 4; i++) Y2[i] = T9[i] ^ T10[i];
    print_hex("Y2", Y2);

    // Step 11: Copy Z2t → Z2
    for (int i = 0; i < 4; i++) Z2[i] = Z2t[i];
    print_hex("Z2", Z2);

    printf("=== End Point Addition ===\n");
}

                                  
// ============================================================
// Itoh–Tsujii inversion µ-sequence for GF(2^233)
// ============================================================
__device__ void gf2_inv_itoh_tsujii_233(uint64_t *r, const uint64_t *a) {
    uint64_t t1[4], t2[4], t3[4];
    uint64_t b1[4] = {a[0], a[1], a[2], a[3]};
    uint64_t b2[4], b4[4], b8[4], b16[4], b32[4], b64[4], b128[4], b192[4], b224[4], b232[4];

    // β2 = (β1)^2 * β1
    gf2_square(t1, b1);
    gf2_mult_233(b2, t1, b1);

    // β4 = (β2)^(2^2) * β2
    gf2_square_k(t1, b2, 2);
    gf2_mult_233(b4, t1, b2);

    // β8 = (β4)^(2^4) * β4
    gf2_square_k(t1, b4, 4);
    gf2_mult_233(b8, t1, b4);

    // β16 = (β8)^(2^8) * β8
    gf2_square_k(t1, b8, 8);
    gf2_mult_233(b16, t1, b8);

    // β32 = (β16)^(2^16) * β16
    gf2_square_k(t1, b16, 16);
    gf2_mult_233(b32, t1, b16);

    // β64 = (β32)^(2^32) * β32
    gf2_square_k(t1, b32, 32);
    gf2_mult_233(b64, t1, b32);

    // β128 = (β64)^(2^64) * β64
    gf2_square_k(t1, b64, 64);
    gf2_mult_233(b128, t1, b64);

    // β192 = (β128)^(2^64) * β64
    gf2_square_k(t1, b128, 64);
    gf2_mult_233(b192, t1, b64);

    // β224 = (β192)^(2^32) * β32
    gf2_square_k(t1, b192, 32);
    gf2_mult_233(b224, t1, b32);

    // β232 = (β224)^(2^8) * β8
    gf2_square_k(t1, b224, 8);
    gf2_mult_233(b232, t1, b8);

    // inv = (β232)^2
    gf2_square(r, b232);
}


// ============================================================
// Scalar multiplication (Lopez–Dahab double & add)
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

    // ===== find the highest set bit =====
    int msb = -1;
    for (int i = 233 - 1; i >= 0; --i) {
        if (get_bit(k, i)) { msb = i; break; }
    }
    if (msb < 0) { printf("k = 0\n"); return; }

    printf("\n=== Scalar Multiplication (GF(2^233)) ===\n");
    printf("MSB = %d\n", msb);
    printf("----------------------------------------\n");

    // ===== main loop (skip first MSB bit) =====
    for (int i = msb - 1; i >= 0; --i) {
        printf("[bit %d] Doubling\n", i);
        point_double_233(X2, Y2, Z2, X, Y, Z, a_curve, b_curve);

        if (get_bit(k, i)) {
            printf("[bit %d] Addition triggered\n", i);
            point_add_233(X2, Y2, Z2, X2, Y2, Z2, Px, Py, a_curve);
        }

        for (int j = 0; j < 4; j++) {
            X[j] = X2[j];
            Y[j] = Y2[j];
            Z[j] = Z2[j];
        }
    }

    // ===== projective → affine (unchanged) =====
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
cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
include_paths = [f"{cuda_root}\\include", f"{cuda_root}\\include\\crt"]
options = ('--std=c++11',) + tuple(f'-I{p}' for p in include_paths)
mod = cp.RawModule(code=kernel_code, options=options)
scalar_kernel = mod.get_function("scalar_mult_233")

Px, Py, Pz = map(cp.asarray, (Gx, Gy, Gz))
a_gpu, b_gpu = map(cp.asarray, (a_curve, b_curve))
Xout, Yout, Zout = (cp.zeros(4, dtype=cp.uint64) for _ in range(3))

num_tests = 15
for t in range(num_tests):
    k_int = random.getrandbits(M - 1) | (1 << (M - 1))
    k_limbs = to_limbs(k_int)
    k_gpu = cp.asarray(k_limbs, dtype=cp.uint64)

    start = time.perf_counter()
    scalar_kernel((1,), (1,), (Px, Py, Pz, a_gpu, b_gpu, k_gpu, Xout, Yout, Zout))
    cp.cuda.Device().synchronize()
    gpu_time = (time.perf_counter() - start) * 1000

    Xr = (int(Xout[3].get()) << 192) | (int(Xout[2].get()) << 128) | (int(Xout[1].get()) << 64) | int(Xout[0].get())
    Yr = (int(Yout[3].get()) << 192) | (int(Yout[2].get()) << 128) | (int(Yout[1].get()) << 64) | int(Yout[0].get())

    print("===================================================")
    print(f"Test #{t+1}")
    print(f"k_int   = 0x{k_int:059x}")
    print(f"k_limbs = [{hex(k_limbs[3])}, {hex(k_limbs[2])}, {hex(k_limbs[1])}, {hex(k_limbs[0])}]")
    print(f"GPU runtime = {gpu_time:.2f} ms")
    print(f"x_aff = 0x{Xr:059x}")
    print(f"y_aff = 0x{Yr:059x}")
    print("===================================================")
