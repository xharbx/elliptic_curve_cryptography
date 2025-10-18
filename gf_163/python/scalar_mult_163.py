# -*- coding: utf-8 -*-
"""
Lopez–Dahab scalar multiplication in GF(2^163) (sect163r2 curve).
Equation: y^2 + xy = x^3 + ax^2 + b
a = 1
b = 0x20A601907B8C953CA1481EB10512F78744A3205FD
"""

import random

# Field parameters
M = 163
IRRED_POLY = (1 << 163) | (1 << 7) | (1 << 6) | (1 << 3) | 1
a_curve = 1
b_curve = 0x20A601907B8C953CA1481EB10512F78744A3205FD

# Base point G
Gx = 0x3f0eba16286a2d57ea0991168d4994637e8343e36
Gy = 0x0d51fbc6c71a0094fa2cdd545b11c5c0c797324f1

# Counters
mult_count = 0
sqr_count = 0


# ============================================================
# GF(2^m) arithmetic
# ============================================================


def gf2m_add(a, b):
    return a ^ b  # XOR in GF(2)

def gf2m_reduce(x, poly=IRRED_POLY, m=M):
    while x.bit_length() > m:
        shift = x.bit_length() - m - 1
        x ^= poly << shift
    return x


def gf2m_mult(a, b, poly=IRRED_POLY, m=M, label=""):
    global mult_count
    result = 0
    bb = b
    aa = a
    while bb:
        if bb & 1:
            result ^= aa
        bb >>= 1
        aa <<= 1
        if aa >> m:
            aa ^= poly
    result = gf2m_reduce(result, poly, m)
    mult_count += 1
    if label:
        print(f"[M{mult_count}] {label} = {hex(result)}")
    return result


def gf2m_square(a, poly=IRRED_POLY, m=M, label=""):
    global sqr_count
    sqr_count += 1
    x = 0
    for i in range(a.bit_length()):
        if (a >> i) & 1:
            x ^= (1 << (2 * i))
    r = gf2m_reduce(x, poly, m)
    if label:
        print(f"[S{sqr_count}] {label} = {hex(r)}")
    return r


def gf2m_square_k(a, k, poly=IRRED_POLY, m=M, base_label=""):
    """Square k times and print each step"""
    r = a
    for i in range(1, k + 1):
        r = gf2m_square(r, poly, m, label=f"{base_label}^(2^{i})")
    return r


def itoh_tsujii_inverse_chain(a, poly=IRRED_POLY, m=M):
    if a == 0:
        raise ZeroDivisionError("No inverse for 0 in GF(2^m)")

    global mult_count, sqr_count
    mult_count = 0
    sqr_count = 0

    print("==== Itoh–Tsujii µ-sequence for m=163 ====")

    # Step 1: β1 = a
    b1 = a
    print("β1 =", hex(b1))

    # Step 2: β2 = (β1)^2 * β1
    t = gf2m_square(b1, label="β1^2")
    b2 = gf2m_mult(t, b1, label="β2")
    
    # Step 3: β3 = (β2)^2 * β1
    t = gf2m_square(b2, label="β2^2")
    b3 = gf2m_mult(t, b1, label="β3")

    # Step 4: β5 = (β3)^(2^2) * β2
    t = gf2m_square_k(b3, 2, base_label="β3")
    b5 = gf2m_mult(t, b2, label="β5")

    # Step 5: β10 = (β5)^(2^5) * β5
    t = gf2m_square_k(b5, 5, base_label="β5")
    b10 = gf2m_mult(t, b5, label="β10")

    # Step 6: β20 = (β10)^(2^10) * β10
    t = gf2m_square_k(b10, 10, base_label="β10")
    b20 = gf2m_mult(t, b10, label="β20")

    # Step 7: β40 = (β20)^(2^20) * β20
    t = gf2m_square_k(b20, 20, base_label="β20")
    b40 = gf2m_mult(t, b20, label="β40")

    # Step 8: β80 = (β40)^(2^40) * β40
    t = gf2m_square_k(b40, 40, base_label="β40")
    b80 = gf2m_mult(t, b40, label="β80")

    # Step 9: β81 = (β80)^(2^1) * β1
    t = gf2m_square(b80, label="β80^2")
    b81 = gf2m_mult(t, b1, label="β81")

    # Step 10: β162 = (β81)^(2^81) * β81
    t = gf2m_square_k(b81, 81, base_label="β81")
    b162 = gf2m_mult(t, b81, label="β162")

    # Step 11: Final inverse = (β162)^2
    inv_a = gf2m_square(b162, label="inv")

    print("=========================================")
    print(f"Multiplications = {mult_count}")
    print(f"Squarings       = {sqr_count}")

    return inv_a

# ============================================================
# Lopez–Dahab Projective Point Operations
# ============================================================
# Precompute c = b^(2^(m-1))
def precompute_c(b, m, poly=IRRED_POLY):
    c = b
    for _ in range(m-1):
        c = gf2m_square(c, poly, m)
    return c



def point_double(P,x):
    """
    Lopez–Dahab projective point doubling for GF(2^163).
    Curve: y^2 + xy = x^3 + ax^2 + b
    Optimized algorithm: 3 multiplications, 5 squarings.

    Input:  P = (X1, Y1, Z1)
    Output: (X2, Y2, Z2) = 2P
    """
    X1, Y1, Z1 = P
    if Z1 == 0:
        return (1, 0, 0)  # point at infinity

    # Step 1: compute squares (5 squarings)
    X1_sq  = gf2m_square(X1)           # X1^2
    Z1_sq  = gf2m_square(Z1)           # Z1^2
    Z1_4   = gf2m_square(Z1_sq)        # Z1^4
    X1_4   = gf2m_square(X1_sq)        # X1^4
    Y1_sq  = gf2m_square(Y1)           # Y1^2

    # Step 2: Z2 = X1^2 * Z1^2   (1 mult)
    Z2 = gf2m_mult(X1_sq, Z1_sq)

    # Step 3: X2 = X1^4 + b*Z1^4   (1 mult + 1 add)
    bZ1_4 = gf2m_mult(b_curve, Z1_4)   # b*Z1^4
    X2    = gf2m_add(X1_4, bZ1_4)

    # Step 4: Y2 = (b*Z1^4)*Z2 + X2 * (a*Z2 + Y1^2 + b*Z1^4)
    left   = gf2m_mult(bZ1_4, Z2)      # (b*Z1^4)*Z2   (2nd mult)
    aZ2    = gf2m_mult(a_curve, Z2)    # a*Z2
    inner  = gf2m_add(aZ2, Y1_sq)
    right  = gf2m_add(inner, bZ1_4)
    X2right = gf2m_mult(X2, right)     # X2 * (...)   (3rd mult)
    Y2     = gf2m_add(left, X2right)

    return (X2, Y2, Z2)
 

def point_add(P, Q):
    """
    Lopez–Dahab projective point addition for GF(2^163).
    Optimized for the special case Z1 = 1.

    Curve: y^2 + xy = x^3 + ax^2 + b
    Input : P = (X0, Y0, Z0), Q = (X1, Y1, 1)
    Output: (X2, Y2, Z2) = P + Q
    """

    X0, Y0, Z0 = P
    print(f"X0 = {hex(X0)}")
    print(f"Y0 = {hex(Y0)}")
    print(f"Z0 = {hex(Z0)}")

    X1, Y1, Z1 = Q
    print(f"X1 = {hex(X1)}")
    print(f"Y1 = {hex(Y1)}")
    print(f"Z1 = {hex(Z1)}")

    assert Z1 == 1, "This optimized routine only works when Z1 = 1"

    # Step 1: A = Y1 * Z0^2 + Y0
    T1 = gf2m_square(Z0)              # Z0^2
    print("T1 (Z0^2) =", hex(T1))
    
    T2 = gf2m_mult(Y1, T1)            # Y1 * Z0^2
    print("T2 (Y1*Z0^2) =", hex(T2))
    
    A  = gf2m_add(T2, Y0)
    print("A =", hex(A))
    
    # Step 2: B = X1 * Z0 + X0
    T3 = gf2m_mult(X1, Z0)
    print("T3 (X1*Z0) =", hex(T3))
    
    B  = gf2m_add(T3, X0)
    print("B =", hex(B))
    
    # Step 3: C = Z0 * B
    C  = gf2m_mult(Z0, B)
    print("C =", hex(C))
    
    # Step 4: D = B^2 * (C + a * Z0^2)
    B2 = gf2m_square(B)
    print("B2 (B^2) =", hex(B2))
    
    T4 = gf2m_mult(a_curve, T1)       # a * Z0^2
    print("T4 (a*Z0^2) =", hex(T4))
    
    T5 = gf2m_add(C, T4)
    print("T5 (C+a*Z0^2) =", hex(T5))
    
    D  = gf2m_mult(B2, T5)
    print("D =", hex(D))
    
    # Step 5: Z2 = C^2
    Z2 = gf2m_square(C)
    print("Z2 =", hex(Z2))
    
    # Step 6: E = A * C
    E  = gf2m_mult(A, C)
    print("E =", hex(E))
    
    # Step 7: X2 = A^2 + D + E
    A2 = gf2m_square(A)
    print("A2 (A^2) =", hex(A2))
    
    T6 = gf2m_add(A2, D)
    print("T6 (A^2+D) =", hex(T6))
    
    X2 = gf2m_add(T6, E)
    print("X2 =", hex(X2))
    
    # Step 8: F = X2 + X1 * Z2
    T7 = gf2m_mult(X1, Z2)
    print("T7 (X1*Z2) =", hex(T7))
    
    F  = gf2m_add(X2, T7)
    print("F =", hex(F))
    
    # Step 9: G = X2 + Y1 * Z2
    T8 = gf2m_mult(Y1, Z2)
    print("T8 (Y1*Z2) =", hex(T8))
    
    G  = gf2m_add(X2, T8)
    print("G =", hex(G))
    
    # Step 10: Y2 = E * F + Z2 * G
    T9 = gf2m_mult(E, F)
    print("T9 (E*F) =", hex(T9))
    
    T10 = gf2m_mult(Z2, G)
    print("T10 (Z2*G) =", hex(T10))
    
    Y2  = gf2m_add(T9, T10)
    print("Y2 =", hex(Y2))

    return (X2, Y2, Z2)


def to_affine(P):
    X, Y, Z = P
    print(f"CONVER- X = {hex(X)}")
    print(f"CONVER- Y = {hex(Y)}")
    print(f"CONVER- Z = {hex(Z)}")
    
    if Z == 0:
        return (0, 0)
    Zinv = itoh_tsujii_inverse_chain(Z)
    x = gf2m_mult(X, Zinv)
    print(f" gf2m_mult x  = {hex(x)}")
    y = gf2m_mult(Y, gf2m_square(Zinv))
    print(f" gf2m_mult y  = {hex(y)}")

    return (x, y)

# ============================================================
# Scalar multiplication
# ============================================================
def scalar_mult(k, P, c_const):
    Q = P  # start from P, not from infinity
    for i in reversed(range(k.bit_length() - 1)):  # skip MSB
        Q = point_double(Q, c_const)
        x1, y1 = to_affine(Q)
        print(f"x1 = {hex(x1)}")
        print(f"y1 = {hex(y1)}")
        
        X0, Y0, Z0 = Q
        print(f"DOUBLE- X0 = {hex(X0)}")
        print(f"DOUBLE- Y0 = {hex(Y0)}")
        print(f"DOUBLE- Z0 = {hex(Z0)}")

        
        if (k >> i) & 1:
            Q = point_add(Q, P)
            X0, Y0, Z0 = Q
            print(f"ADD- X0 = {hex(X0)}")
            print(f"ADD- Y0 = {hex(Y0)}")
            print(f"ADD- Z0 = {hex(Z0)}")
            
            x1, y1 = to_affine(Q)
            print(f"x1 = {hex(x1)}")
            print(f"y1 = {hex(y1)}")
    return Q

# ============================================================
# Test with NIST known values
# ============================================================

def ecc_gf2m_163(k, P):
    """
    Perform scalar multiplication k*P on sect163r2 over GF(2^163).
    Arguments:
        k (int): scalar
        P (tuple): point in projective coordinates (X, Y, Z)
    Returns:
        (x, y): affine coordinates
    """

    # Precomputation
    c_const = precompute_c(b_curve, M)

    # Scalar multiplication
    R = scalar_mult(k, P, c_const)

    # Convert back to affine
    x2, y2 = to_affine(R)

    # Debug prints
    print(f"x2 = {hex(x2)}")
    print(f"y2 = {hex(y2)}")

    X1, Y1, Z1 = R
    print(f"SCALER- X1 = {hex(X1)}")
    print(f"SCALER- Y1 = {hex(Y1)}")
    print(f"SCALER- Z1 = {hex(Z1)}")

    return x2, y2


# ============================================================
# Example usage with NIST base point
# ============================================================
import time

if __name__ == "__main__":
    G = (Gx, Gy, 1)  # base point in projective form
    k = 0x7fff705303b81f7440ecd05e56f02d89db3d00e14
    print(f"Scalar k = {hex(k)}")

    # ---- Start timer ----
    start_time = time.perf_counter()

    P_out = ecc_gf2m_163(k, G)

    # ---- End timer ----
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    print(f"\nScalar multiplication completed in {elapsed_ms:.3f} ms")

    X, Y = P_out
    print("\n=== Result (Projective Coordinates) ===")
    print(f"X = 0x{X:041x}")
    print(f"Y = 0x{Y:041x}")

