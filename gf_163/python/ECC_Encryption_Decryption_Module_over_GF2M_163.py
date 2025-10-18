# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 06:49:00 2025

@author: xsala
"""

import random
from scalar_mult_163 import ecc_gf2m_163, point_add, to_affine, Gx, Gy
from generating_points_ecc_163 import generate_scalar_points

# === Curve order for sect163r2 ===
n = 0x40000000000000000000292FE77E70C12A4234C33
G = (Gx, Gy, 1)  # Base point (projective coordinates)

# ----------------------------
# Key generation (Bob)
# ----------------------------
def generate_keys():
    d = random.randrange(1, n)       # Private key (scalar)
    e1 = G                           # Base point
    # Multiply to get public key
    e2_proj = ecc_gf2m_163(d, e1)    # (X, Y, Z) possibly projective
    # Normalize: set Z = 1 if your code expects 3-tuple
    e2 = (e2_proj[0], e2_proj[1], 1)

    return d, e1, e2                # (private, base, public)

# ----------------------------
# ECC ElGamal Encryption
# ----------------------------
def encrypt(P, e1, e2):
    r = random.randrange(1, n)
    r_enc = r
    # Ensure all points have (x, y, 1)
    P_z = (P[0], P[1], 1)
    e1 = (e1[0], e1[1], 1)
    e2 = (e2[0], e2[1], 1)

    # Compute ciphertext points
    C1 = ecc_gf2m_163(r, e1)          # C1 = r * e1
    rE2 = ecc_gf2m_163(r, e2)         # rE2 = r * e2
    rE2_z= (rE2[0], rE2[1], 1)
    C2_z = point_add(P_z, rE2_z)            # C2 = P + r * e2
    C2 = to_affine(C2_z)

    return r_enc, C1, C2

# ----------------------------
# ECC ElGamal Decryption
# ----------------------------
def decrypt(C1, C2, d):
    """
    Decrypts the ciphertext (C1, C2) using private key d.
    Returns the recovered plaintext point P (x, y).
    """
    # Ensure inputs are in (x, y, 1)
    C1_z = (C1[0], C1[1], 1)
    C2_z = (C2[0], C2[1], 1)

    # Compute shared secret: d * C1
    dC1 = ecc_gf2m_163(d, C1_z)

    # Negate shared secret (for subtraction)
    def point_neg(P):
        x, y = P
        return (x, x ^ y)
    
    neg_dC1 = point_neg(dC1)
    neg_dC1_z = (neg_dC1[0], neg_dC1[1], 1)


    # Recover plaintext:  P = C2 - d*C1  =  C2 + (-d*C1)
    P_rec_z = point_add(C2_z, neg_dC1_z)
    P_rec = to_affine(P_rec_z)

    return P_rec


# ----------------------------
# Example Run
# ----------------------------
if __name__ == "__main__":
    # Key generation
    d, e1, e2 = generate_keys()

    # ----------------------------
    # Message point P
    # ----------------------------
    P = (
        0x4547BD66270DF7A9601351A616FEF080D44528B03,
        0x19303302D63359036B047497DC2F1BB94BB3D93C4
    )

    # ----------------------------
    # Encrypt point P
    # ----------------------------
    r_enc, C1, C2 = encrypt(P, e1, e2)

    # ----------------------------
    # Decrypt points C1, C2
    # ----------------------------
    P_dec = decrypt(C1, C2, d)

    # ----------------------------
    # Print all results
    # ----------------------------
    print("\n=== Key Generation ===")
    print(f"Private key (d): {hex(d)}")

    print("\nPublic key e1 (base point):")
    print(f"  x = {hex(e1[0])}")
    print(f"  y = {hex(e1[1])}")

    print("\nPublic key e2 (d * e1):")
    print(f"  x = {hex(e2[0])}")
    print(f"  y = {hex(e2[1])}")

    print("\n=== Message Point ===")
    print("Original Message P:")
    print(f"  x = {hex(P[0])}")
    print(f"  y = {hex(P[1])}")

    print("\n=== Ciphertext ===")
    print(f"Random number (r): {hex(r_enc)}")
    print("C1:")
    print(f"  x = {hex(C1[0])}")
    print(f"  y = {hex(C1[1])}")
    print("C2:")
    print(f"  x = {hex(C2[0])}")
    print(f"  y = {hex(C2[1])}")

    print("\n=== Decryption Result ===")
    print("Recovered P:")
    print(f"  x = {hex(P_dec[0])}")
    print(f"  y = {hex(P_dec[1])}")

# ============================================================
# Step 1. Generate alphabet dictionary A–Z → ECC Points
# ============================================================

def build_alphabet_dict():
    points = generate_scalar_points(26)
    alphabet = {}
    for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        alphabet[ch] = points[i]
    return alphabet

# ============================================================
# Step 2. ECC Text Encryption / Decryption
# ============================================================

def encrypt_message(msg, e1, e2, alphabet):
    ciphertext = []
    for ch in msg.upper():
        if ch not in alphabet:
            continue

        x_val, y_val = alphabet[ch]

        # convert only if strings (hex form)
        if isinstance(x_val, str):
            P = (int(x_val, 16), int(y_val, 16))
        else:
            P = (x_val, y_val)

        r_enc, C1, C2 = encrypt(P, e1, e2)
        ciphertext.append((C1, C2))

    return ciphertext


def decrypt_message(ciphertext, d, alphabet):
    # Build reverse lookup table safely
    reverse_map = {}
    for k, v in alphabet.items():
        x_val, y_val = v
        if isinstance(x_val, str):
            reverse_map[(int(x_val, 16), int(y_val, 16))] = k
        else:
            reverse_map[(x_val, y_val)] = k

    # Start decryption
    plaintext = ""
    for C1, C2 in ciphertext:
        P_dec = decrypt(C1, C2, d)
        P_aff = (P_dec[0], P_dec[1])
        plaintext += reverse_map.get(P_aff, "?")

    return plaintext
# ============================================================
# Step 3. Demo Run
# ============================================================

if __name__ == "__main__":
    # Generate ECC key pair
    d, e1, e2 = generate_keys()

    # Build alphabet lookup
    alphabet = build_alphabet_dict()

    # Message to encrypt
    message = "SALAH"

    # Encrypt message
    ciphertext = encrypt_message(message, e1, e2, alphabet)


    # Decrypt message
    decrypted = decrypt_message(ciphertext, d, alphabet)


    print("\n=== Original Message ===")
    print(message)


    print("\n=== Ciphertext (C1, C2 pairs) ===")
    for i, (C1, C2) in enumerate(ciphertext):
        print(f"\nLetter {message[i]}:")
        print(f"C1 = ({hex(C1[0])}, {hex(C1[1])})")
        print(f"C2 = ({hex(C2[0])}, {hex(C2[1])})")


    print("\n=== Decrypted Message ===")
    print(decrypted)
    
    
    