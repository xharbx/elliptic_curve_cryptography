# -*- coding: utf-8 -*-
"""
Elliptic Curve Digital Signature Scheme (ECDSS) over GF(2^233)
Lopez–Dahab projective coordinates, sect233r1 parameters.
"""

import random
import hashlib
import time
import matplotlib.pyplot as plt
from scalar_mult_233 import ecc_gf2m_233, point_add, to_affine, Gx, Gy
from ECC_Encryption_Decryption_Module_over_GF2M_233 import generate_keys, n

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
        P = ecc_gf2m_233(r, G)
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

    A_G = ecc_gf2m_233(A, G)
    A_G_z = (A_G[0], A_G[1], 1)
    B_Q = ecc_gf2m_233(B, e2_z)
    B_Q_z = (B_Q[0], B_Q[1], 1)

    T = point_add(A_G_z, B_Q_z)
    xT, yT = to_affine(T)

    return (xT % n) == S1

# === Demo Run ===
if __name__ == "__main__":
    num_runs = 5
    times = []

    print("\n" + "="*60)
    print("Running ECDSA Signing & Verification 5 Times")
    print("="*60)

    # Generate keys once
    d, e1, e2 = generate_keys()
    msg = "ECC SIGNATURE SALAH"

    print("\n=== Keys ===")
    print(f"Private key d = {hex(d)}")
    print(f"Public key e2 = ({hex(e2[0])}, {hex(e2[1])})")
    print("\n" + "="*60)

    # Run the signing and verification process 5 times
    for i in range(1, num_runs + 1):
        print(f"\n--- Run {i} ---")

        # Start timing
        start_time = time.time()

        # Sign the message
        S1, S2 = sign_message(msg, d)

        # Verify the signature
        valid = verify_signature(msg, S1, S2, e2)

        # End timing
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)

        print(f"S1 = {hex(S1)}")
        print(f"S2 = {hex(S2)}")
        print(f"Valid signature: {'PASS' if valid else 'FAIL'}")
        print(f"Time elapsed: {elapsed:.6f} seconds")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\n" + "="*60)
    print("=== Timing Statistics ===")
    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Min time: {min_time:.6f} seconds")
    print(f"Max time: {max_time:.6f} seconds")
    print("="*60)

    # Plot the results
    plt.figure(figsize=(10, 6))
    runs = list(range(1, num_runs + 1))

    plt.plot(runs, times, marker='o', linestyle='-', linewidth=2, markersize=8, color='#2E86AB')
    plt.axhline(y=avg_time, color='red', linestyle='--', linewidth=1.5, label=f'Average: {avg_time:.6f}s')

    plt.xlabel('Run Number', fontsize=12, fontweight='bold')
    plt.ylabel('Time Elapsed (seconds)', fontsize=12, fontweight='bold')
    plt.title('ECDSA Signing & Verification Performance\nGF(2^233) - sect233r1', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(runs)

    # Add value labels on points
    for i, (run, t) in enumerate(zip(runs, times)):
        plt.text(run, t, f'{t:.4f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save the figure
    output_file = 'ecdsa_performance_233.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Plot saved as: {output_file}")

    # plt.show()  # Commented out to avoid blocking execution
