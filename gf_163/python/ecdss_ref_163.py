# -*- coding: utf-8 -*-
"""
Elliptic Curve Digital Signature Scheme (ECDSS) over GF(2^163)
Lopez–Dahab projective coordinates, sect163r2 parameters.
"""

import random
import hashlib
from scalar_mult_163 import ecc_gf2m_163, point_add, to_affine, Gx, Gy
from ECC_Encryption_Decryption_Module_over_GF2M_163 import generate_keys, n

# === Modular inverse (for mod n) ===
def modinv(a, m):
    return pow(a, -1, m)

# === Hash message to integer ===
def hash_message(msg):
    h = hashlib.sha256(msg.encode()).hexdigest()
    return int(h, 16) % n

# === ECDSS Signing ===
def sign_message(msg, d):
    hM = hash_message(msg)
    while True:
        r = random.randrange(1, n)
        G = (Gx, Gy, 1)
        P = ecc_gf2m_163(r, G)
        x1, y1 = P
        S1 = x1 % n
        if S1 == 0:
            continue
        r_inv = modinv(r, n)
        S2 = (r_inv * (hM + d * S1)) % n
        if S2 != 0:
            break
    return (S1, S2)

# === ECDSS Verification ===
def verify_signature(msg, S1, S2, e2):
    if not (1 <= S1 < n and 1 <= S2 < n):
        return False

    hM = hash_message(msg)
    w = modinv(S2, n)
    A = (hM * w) % n
    B = (S1 * w) % n

    G = (Gx, Gy, 1)
    e2_z = (e2[0], e2[1], 1)

    A_G = ecc_gf2m_163(A, G)
    A_G_z = (A_G[0], A_G[1], 1)
    B_Q = ecc_gf2m_163(B, e2_z)
    B_Q_z = (B_Q[0], B_Q[1], 1)

    T = point_add(A_G_z, B_Q_z)
    xT, yT = to_affine(T)

    return (xT % n) == S1

# === Demo Run ===
if __name__ == "__main__":
    d, e1, e2 = generate_keys()
    msg = "ECC SIGNATURE SALAH"

    print("\n=== Keys ===")
    print(f"Private key d = {hex(d)}")
    print(f"Public key e2 = ({hex(e2[0])}, {hex(e2[1])})")

    S1, S2 = sign_message(msg, d)
    print("\n=== Signature ===")
    print(f"S1 = {hex(S1)}")
    print(f"S2 = {hex(S2)}")

    valid = verify_signature(msg, S1, S2, e2)
    print("\n=== Verification ===")
    print("Valid signature ✅" if valid else "Invalid ❌")
