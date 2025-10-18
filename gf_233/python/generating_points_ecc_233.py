# -*- coding: utf-8 -*-
"""
Generate ECC Points on sect233r1 using Lopez–Dahab scalar multiplication
Each point = k * G for k = 1, 2, 3, ...
"""

from scalar_mult_233 import (
    ecc_gf2m_233,
    point_add,
    point_double,
    to_affine,
    Gx,
    Gy,
)

# ============================================================
# Generate Scalar Multiples of Base Point
# ============================================================

def generate_scalar_points(num_points):
    """
    Generate the first num_points ECC points using scalar multiplication:
        P_k = k * G
    """
    G = (Gx, Gy, 1)
    points = []

    for k in range(1, num_points + 1):
        P = ecc_gf2m_233(k, G)
        x_hex = hex(P[0])
        y_hex = hex(P[1])
        points.append(P)
    return points

# ============================================================
# Example Run
# ============================================================

if __name__ == "__main__":
    points = generate_scalar_points(26)
    print("\n=== Summary of Generated Points (Hex) ===\n")
    for i, (x_hex, y_hex) in enumerate(points, start=1):
        print(f"P{i:02d}:")
        print(f"  x = {hex(x_hex)}")
        print(f"  y = {hex(y_hex)}\n")
