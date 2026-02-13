# -*- coding: utf-8 -*-
"""
GPU Lopez–Dahab Scalar Multiplication over GF(2^163)
Curve: y^2 + xy = x^3 + x^2 + b
f(x) = x^163 + x^7 + x^6 + x^3 + 1
Author: Salah Harb
"""

import cupy as cp
import numpy as np
import time

# ============================================================
# Parameters for sect163r2
# ============================================================
M = 163
IRRED_POLY = (1 << 163) | (1 << 7) | (1 << 6) | (1 << 3) | 1

a_curve_int = 1
b_curve_int = 0x20A601907B8C953CA1481EB10512F78744A3205FD

Gx_int = 0x03f0eba16286a2d57ea0991168d4994637e8343e36
Gy_int = 0x0d51fbc6c71a0094fa2cdd545b11c5c0c797324f1
Gz_int = 1

# ============================================================
# Convert to limb arrays (3 × 64-bit)
# ============================================================
def to_limbs(x):
    return np.array([
        np.uint64(x & ((1 << 64) - 1)),
        np.uint64((x >> 64) & ((1 << 64) - 1)),
        np.uint64(x >> 128)
    ], dtype=np.uint64)

Gx = to_limbs(Gx_int)
Gy = to_limbs(Gy_int)
Gz = to_limbs(Gz_int)
a_curve = to_limbs(a_curve_int)
b_curve = to_limbs(b_curve_int)

# ============================================================
# Display base point (full 163-bit)
# ============================================================
print("===================================================")
print("Base Point P (sect163r2):")
print(f"Gx = 0x{Gx_int:041x}")
print(f"Gy = 0x{Gy_int:041x}")
print(f"Gz = 0x{Gz_int:041x}")
print("===================================================")

# ============================================================
# GPU Kernel Code
# ============================================================
kernel_code = r'''
#ifndef __UINT64_T_DEFINED
typedef unsigned long long uint64_t;
#define __UINT64_T_DEFINED
#endif

extern "C" {

// ============================================================
// GF(2^163) Field Arithmetic
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
// Squaring
// ============================================================
__device__ void gf2_square(uint64_t *r, const uint64_t *a) {
    uint64_t t[6] = {0};

    for (int limb = 0; limb < 3; limb++) {
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

    // Reduce
    for (int bit = 383; bit >= 163; --bit) {
        int limb = bit / 64;
        int pos  = bit % 64;
        if (t[limb] & (1ULL << pos)) {
            t[limb] ^= (1ULL << pos);
            int shift = bit - 163;
            int targets[4] = {shift + 7, shift + 6, shift + 3, shift};
            for (int k = 0; k < 4; k++) {
                int tgt = targets[k];
                if (tgt < 0) continue;
                int w = tgt / 64, bpos = tgt % 64;
                t[w] ^= (1ULL << bpos);
            }
        }
    }

    r[0] = t[0];
    r[1] = t[1];
    r[2] = t[2] & ((1ULL << 35) - 1);
}

// ============================================================
// k-fold squaring
// ============================================================
__device__ void gf2_square_k(uint64_t *r, const uint64_t *a, int k) {
    uint64_t tmp1[3], tmp2[3];
    tmp1[0] = a[0]; tmp1[1] = a[1]; tmp1[2] = a[2];
    for (int i = 0; i < k; i++) {
        gf2_square(tmp2, tmp1);
        tmp1[0] = tmp2[0]; tmp1[1] = tmp2[1]; tmp1[2] = tmp2[2];
    }
    r[0] = tmp1[0]; r[1] = tmp1[1]; r[2] = tmp1[2];
}

                                        
// ============================================================
// XOR combine 6-limb temporaries t[] ← t[] ⊕ (val << shift)
// ============================================================
__device__ void xor_shift(uint64_t *t, const uint64_t *val,
                          int nval, int shift) {
    int w = shift / 64;
    int b = shift % 64;
    for (int i = 0; i < nval; i++) {
        t[w + i] ^= val[i] << b;
        if (b && (w + i + 1 < 6))
            t[w + i + 1] ^= val[i] >> (64 - b);
    }
}

// ============================================================
// Karatsuba 163-bit carryless multiply (3×3 limb)
// ============================================================
__device__ void gf2_mult_163(uint64_t *r, const uint64_t *a,
                             const uint64_t *b) {
    uint64_t t[6] = {0};

    // 3×3 schoolbook — safe baseline for correctness
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            uint64_t hi, lo;
            gf2_mul64(&hi, &lo, a[i], b[j]);
            xor_shift(t, &lo, 1, (i + j) * 64);
            xor_shift(t, &hi, 1, (i + j) * 64 + 64);
        }
    }

    // Modular reduction mod x^163 + x^7 + x^6 + x^3 + 1
    for (int bit = 383; bit >= 163; --bit) {
        int limb = bit / 64;
        int pos = bit % 64;
        if (t[limb] & (1ULL << pos)) {
            t[limb] ^= (1ULL << pos);
            int shift = bit - 163;
            int targets[4] = {shift + 7, shift + 6, shift + 3, shift};
            for (int k = 0; k < 4; k++) {
                int tgt = targets[k];
                if (tgt < 0) continue;
                int w = tgt / 64, bpos = tgt % 64;
                t[w] ^= (1ULL << bpos);
            }
        }
    }

    r[0] = t[0];
    r[1] = t[1];
    r[2] = t[2] & ((1ULL << 35) - 1);
}
                                
// ============================================================
// Print helper
// ============================================================

__device__ void print_hex(const char* name, const uint64_t *v) {
    printf("%s = 0x%016llx%016llx%016llx\n", name, v[2], v[1], v[0]);
}

// ============================================================
// Lopez–Dahab Point Double
// ============================================================ 

__device__ void point_double_163(uint64_t *X2,uint64_t *Y2,uint64_t *Z2,
                                 const uint64_t *X1,const uint64_t *Y1,const uint64_t *Z1,
                                 const uint64_t *a_curve,const uint64_t *b_curve){
    uint64_t X1_sq[3],Z1_sq[3],Z1_4[3],X1_4[3],Y1_sq[3];
    uint64_t bZ1_4[3],aZ2[3],inner[3],X2right[3],left[3];


    gf2_square(X1_sq,X1); 
    gf2_square(Z1_sq,Z1); 
    gf2_square(Z1_4,Z1_sq);
    gf2_square(X1_4,X1_sq);
    gf2_square(Y1_sq,Y1);  

    gf2_mult_163(Z2,X1_sq,Z1_sq); 
    gf2_mult_163(bZ1_4,b_curve,Z1_4);

    for(int i=0;i<3;i++) X2[i]=X1_4[i]^bZ1_4[i];


    gf2_mult_163(left,bZ1_4,Z2); 
    gf2_mult_163(aZ2,a_curve,Z2); 
    for(int i=0;i<3;i++) inner[i]=aZ2[i]^Y1_sq[i]^bZ1_4[i];


    gf2_mult_163(X2right,X2,inner); 
    for(int i=0;i<3;i++) Y2[i]=left[i]^X2right[i];
}


// ============================================================
// Lopez–Dahab Point Add (Z1=1)
// ============================================================
__device__ void point_add_163(uint64_t *X2, uint64_t *Y2, uint64_t *Z2,
                              const uint64_t *X0, const uint64_t *Y0, const uint64_t *Z0,
                              const uint64_t *X1, const uint64_t *Y1,
                              const uint64_t *a_curve) {
    uint64_t T1[3], T2[3], A[3], T3[3], B[3], C[3];
    uint64_t B2[3], T4[3], T5[3], D[3], Z2t[3], E[3], A2[3], T6[3], F[3], G[3];
    uint64_t T7[3], T8[3], T9[3], T10[3];

    printf("\n=== Point Addition (GF(2^163)) ===\n");
    print_hex("X0", X0);
    print_hex("Y0", Y0);
    print_hex("Z0", Z0);
    print_hex("X1", X1);
    print_hex("Y1", Y1);
    printf("----------------------------------\n");

    // Step 1: A = Y1 * Z0^2 + Y0
    gf2_square(T1, Z0);
    print_hex("T1 (Z0^2)", T1);

    gf2_mult_163(T2, Y1, T1);
    print_hex("T2 (Y1*Z0^2)", T2);

    for (int i = 0; i < 3; i++) A[i] = T2[i] ^ Y0[i];
    print_hex("A", A);

    // Step 2: B = X1 * Z0 + X0
    gf2_mult_163(T3, X1, Z0);
    print_hex("T3 (X1*Z0)", T3);

    for (int i = 0; i < 3; i++) B[i] = T3[i] ^ X0[i];
    print_hex("B", B);

    // Step 3: C = Z0 * B
    gf2_mult_163(C, Z0, B);
    print_hex("C (Z0*B)", C);

    // Step 4: D = B^2 * (C + a * Z0^2)
    gf2_square(B2, B);
    print_hex("B2 (B^2)", B2);

    gf2_mult_163(T4, a_curve, T1);
    print_hex("T4 (a*Z0^2)", T4);

    for (int i = 0; i < 3; i++) T5[i] = C[i] ^ T4[i];
    print_hex("T5 (C+a*Z0^2)", T5);

    gf2_mult_163(D, B2, T5);
    print_hex("D", D);

    // Step 5: Z2 = C^2
    gf2_square(Z2t, C);
    print_hex("Z2t (C^2)", Z2t);

    // Step 6: E = A * C
    gf2_mult_163(E, A, C);
    print_hex("E (A*C)", E);

    // Step 7: X2 = A^2 + D + E
    gf2_square(A2, A);
    print_hex("A2 (A^2)", A2);

    for (int i = 0; i < 3; i++) T6[i] = A2[i] ^ D[i];
    print_hex("T6 (A^2+D)", T6);

    for (int i = 0; i < 3; i++) X2[i] = T6[i] ^ E[i];
    print_hex("X2", X2);

    // Step 8: F = X2 + X1 * Z2
    gf2_mult_163(T7, X1, Z2t);
    print_hex("T7 (X1*Z2)", T7);

    for (int i = 0; i < 3; i++) F[i] = X2[i] ^ T7[i];
    print_hex("F", F);

    // Step 9: G = X2 + Y1 * Z2
    gf2_mult_163(T8, Y1, Z2t);
    print_hex("T8 (Y1*Z2)", T8);

    for (int i = 0; i < 3; i++) G[i] = X2[i] ^ T8[i];
    print_hex("G", G);

    // Step 10: Y2 = E*F + Z2*G
    gf2_mult_163(T9, E, F);
    print_hex("T9 (E*F)", T9);

    gf2_mult_163(T10, Z2t, G);
    print_hex("T10 (Z2*G)", T10);

    for (int i = 0; i < 3; i++) Y2[i] = T9[i] ^ T10[i];
    print_hex("Y2", Y2);

    // Step 11: Copy Z2t → Z2
    for (int i = 0; i < 3; i++) Z2[i] = Z2t[i];
    print_hex("Z2", Z2);

    printf("=== End Point Addition ===\n");
}

                                  
__device__ void gf2_inv_itoh_tsujii_163(uint64_t *r, const uint64_t *a) {
    uint64_t t1[3], t2[3], t3[3];

    uint64_t b1[3] = {a[0], a[1], a[2]};

    // β2 = (β1)^2 * β1
    gf2_square(t1, b1);
    gf2_mult_163(t2, t1, b1);

    // β3 = (β2)^2 * β1
    gf2_square(t1, t2);
    gf2_mult_163(t3, t1, b1);

    // β5 = (β3)^(2^2) * β2
    gf2_square_k(t1, t3, 2);
    gf2_mult_163(t2, t1, t2);

    // β10 = (β5)^(2^5) * β5
    gf2_square_k(t1, t2, 5);
    gf2_mult_163(t3, t1, t2);

    // β20 = (β10)^(2^10) * β10
    gf2_square_k(t1, t3, 10);
    gf2_mult_163(t2, t1, t3);

    // β40 = (β20)^(2^20) * β20
    gf2_square_k(t1, t2, 20);
    gf2_mult_163(t3, t1, t2);

    // β80 = (β40)^(2^40) * β40
    gf2_square_k(t1, t3, 40);
    gf2_mult_163(t2, t1, t3);

    // β81 = (β80)^(2^1) * β1
    gf2_square(t1, t2);
    gf2_mult_163(t3, t1, b1);

    // β162 = (β81)^(2^81) * β81
    gf2_square_k(t1, t3, 81);
    gf2_mult_163(t2, t1, t3);

    // inv = (β162)^2
    gf2_square(r, t2);
}
                                  
__device__ bool get_bit(const uint64_t *k, int i) {
    int limb = i / 64;
    int bit  = i % 64;
    return (k[limb] >> bit) & 1ULL;
}

// ============================================================
// Scalar multiplication (Lopez–Dahab double & add)
// ============================================================
extern "C" __global__ void scalar_mult_163(
    const uint64_t *Px, const uint64_t *Py, const uint64_t *Pz,
    const uint64_t *a_curve, const uint64_t *b_curve,
    const uint64_t *k,      // <-- now pointer to 3-limb scalar
    uint64_t *Xout, uint64_t *Yout, uint64_t *Zout)
{
    uint64_t X[3] = {Px[0], Px[1], Px[2]};
    uint64_t Y[3] = {Py[0], Py[1], Py[2]};
    uint64_t Z[3] = {Pz[0], Pz[1], Pz[2]};
    uint64_t X2[3], Y2[3], Z2[3];

    // Find MSB (bit 162)
    int msb = 162;
    printf("\n=== Scalar Multiplication (GF(2^163)) ===\n");
    printf("MSB = %d\n", msb);
    printf("k[0-2] = 0x%016llx%016llx%016llx\n", k[2], k[1], k[0]);
    print_hex("Start X", X);
    print_hex("Start Y", Y);
    print_hex("Start Z", Z);
    printf("----------------------------------------\n");

    // Loop through scalar bits
    for (int i = msb - 1; i >= 0; --i) {
        point_double_163(X2, Y2, Z2, X, Y, Z, a_curve, b_curve);

        if (get_bit(k, i)) {
            printf("[bit %d] Addition triggered\n", i);
            point_add_163(X2, Y2, Z2, X2, Y2, Z2, Px, Py, a_curve);
        }

        for (int j = 0; j < 3; j++) {
            X[j] = X2[j];
            Y[j] = Y2[j];
            Z[j] = Z2[j];
        }
    }

    // Output final projective result
    for (int j = 0; j < 3; j++) {
        Xout[j] = X[j];
        Yout[j] = Y[j];
        Zout[j] = Z[j];
    }

    printf("\n=== Final Projective Point ===\n");
    print_hex("Xout", Xout);
    print_hex("Yout", Yout);
    print_hex("Zout", Zout);

    // Convert to affine
    uint64_t Zinv[3], Zinv2[3], x_aff[3], y_aff[3];
    gf2_inv_itoh_tsujii_163(Zinv, Zout);
    gf2_square(Zinv2, Zinv);
    gf2_mult_163(x_aff, Xout, Zinv);
    gf2_mult_163(y_aff, Yout, Zinv2);

    for (int j = 0; j < 3; j++) {
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
scalar_kernel = mod.get_function("scalar_mult_163")

Px = cp.asarray(Gx)
Py = cp.asarray(Gy)
Pz = cp.asarray(Gz)
a_gpu = cp.asarray(a_curve)
b_gpu = cp.asarray(b_curve)

Xout = cp.zeros(3, dtype=cp.uint64)
Yout = cp.zeros(3, dtype=cp.uint64)
Zout = cp.zeros(3, dtype=cp.uint64)

import random
 
# ============================================================
# Run multiple random scalar multiplications (GPU)
# ============================================================
num_tests = 10  # number of random scalars to test

for t in range(num_tests):
    # === Random scalar (163-bit, MSB=1) ===
    k_int = random.getrandbits(M - 1) | (1 << (M - 1))
    k_limbs = [
        (k_int >> 0) & ((1 << 64) - 1),
        (k_int >> 64) & ((1 << 64) - 1),
        (k_int >> 128) & ((1 << 64) - 1)
    ]
    k_gpu = cp.asarray(k_limbs, dtype=cp.uint64)

    # === Launch kernel ===
    start = time.perf_counter()
    scalar_kernel((1,), (1,),
                  (Px, Py, Pz, a_gpu, b_gpu, k_gpu, Xout, Yout, Zout))
    cp.cuda.Device().synchronize()
    gpu_time = (time.perf_counter() - start) * 1000

    # === Collect results ===
    Xr = (int(Xout[2].get()) << 128) | (int(Xout[1].get()) << 64) | int(Xout[0].get())
    Yr = (int(Yout[2].get()) << 128) | (int(Yout[1].get()) << 64) | int(Yout[0].get())

    print("===================================================")
    print(f"Test #{t+1}")
    print("Random scalar k (163-bit):")
    print(f"k_int   = 0x{k_int:041x}")
    print(f"k_limbs = [{hex(k_limbs[2])}, {hex(k_limbs[1])}, {hex(k_limbs[0])}]")
    print("---------------------------------------------------")
    print(f"GPU scalar multiplication time: {gpu_time:.2f} ms")
    print("\n=== Result (Affine Coordinates) ===")
    print(f"x_aff = 0x{Xr:041x}")
    print(f"y_aff = 0x{Yr:041x}")
    print("===================================================\n")

