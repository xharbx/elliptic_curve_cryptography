# Elliptic Curve Cryptography over GF(2^m)

This repository contains complete hardware and software implementations of **Elliptic Curve Cryptography (ECC)** over **NIST binary fields**:

- **GF(2¹⁶³)** → sect163r2  
- **GF(2²³³)** → sect233r1  
- **GF(2⁵⁷¹)** → sect571r1  

Each implementation includes Verilog hardware modules for ECC arithmetic, Python-generated Verilog testbenches, and Python reference implementations for scalar-point multiplication and the Elliptic Curve Digital Signature Scheme (ECDSS).

---

## 📘 Overview

Elliptic Curve Cryptography over binary fields GF(2^m) is widely used in hardware cryptographic accelerators due to its efficient bitwise operations and reduced circuit complexity.  
This project provides **reconfigurable Verilog implementations** of ECC arithmetic and **Python tools** to generate and test them automatically.

All curves use the binary-field Weierstrass equation:

y^2 + xy = x^3 + ax^2 + b

with parameters derived from **NIST FIPS 186-4 (Digital Signature Standard)**.

---

## 📂 Repository Structure

```text
## 📂 Repository Structure

```text
elliptic_curve_cryptography/
│
├── gf2_163/
│   ├── verilog/
│   │   ├── ec_scalar_mult_163.v
│   │   ├── gf2_reduce_163.v
│   │   ├── gf2m_inv_163.v
│   │   ├── gf2m_mult_163.v
│   │   ├── mult20.v
│   │   ├── mult21.v
│   │   ├── mult40.v
│   │   ├── mult41.v
│   │   ├── mult81.v
│   │   ├── mult82.v
│   │   ├── mult163.v
│   │   ├── point_add_ld163.v
│   │   ├── point_double_ld163.v
│   │   └── squarer_163.v
│   │
│   ├── testbench/
│   │   └── tb_ec_scalar_mult_163.v
│   │
│   └── python/
│       ├── scalar_mult_163.py
│       ├── testbench_gen_scalar_mult_163.py
│       ├── generating_points_ecc_163.py
│       ├── ECC_Encryption_Decryption_Module_over_GF2M_163.py
│       └── ecdss_ref_163.py
│
├── gf2_233/
│   ├── verilog/
│   ├── testbench/
│   └── python/
│
├── gf2_571/
│   ├── verilog/
│   ├── testbench/
│   └── python/
│
└── README.md


```

## ⚙️ Implemented Components

### 1. Field Arithmetic

Each finite-field operation is implemented using optimized hardware architectures:

| Operation | Algorithm | Description |
|------------|------------|-------------|
| **Multiplication** | Karatsuba–Ofman | Reduces partial product count for large bit-widths |
| **Inversion** | Itoh–Tsujii | Fast algorithm for computing the multiplicative inverse of an element in a finite field GF(2) |
| **Addition/Subtraction** | XOR | Simple bitwise XOR |

---

### 2. Point Operations (Lopez–Dahab (LD) Projective Coordinates)

All elliptic curve point operations—point addition and point doubling—are implemented using Lopez–Dahab projective coordinates, which remove the need for costly modular inversions in the scalar multiplication loop.

In this representation, an affine point (x, y) is mapped to a projective point (X : Y : Z), where mapping affine point (x, y) to a LD projective point, one sets X = x, Y = y, Z =1.

Now converting the points in LD  projective point to affine point we simply apply: x = X / Z, and y = Y / Z².

The LD eliminates the need for costly field inversions by operating entirely in projective space. It allows all point addition and doubling operations to be performed using only field multiplications, squarings, and additions, making it highly suitable for efficient hardware implementation on FPGA or ASIC platforms.

### References
1. López, J., & Dahab, R. (2002). *Improved Algorithms for Elliptic Curve Arithmetic in GF(2n).*, In: Tavares, S., Meijer, H. (eds) Selected Areas in Cryptography. SAC 1998. Lecture Notes in Computer Science, vol 1556. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-48892-8_16

---

### 3. Scalar Multiplication (k·P)

Implements the **binary double-and-add loop** for scalar multiplication.  
Each iteration performs a point doubling and, conditionally, a point addition.

def scalar_mult(k, P, c_const):
    Q = P  # start from P, not from infinity
    for i in reversed(range(k.bit_length() - 1)):  # skip MSB
        Q = point_double(Q, c_const)
        if (k >> i) & 1:
            Q = point_add(Q, P)
    return Q

This same algorithm is used across GF(2¹⁶³), GF(2²³³), and GF(2⁵⁷¹), producing consistent hardware and simulation results.

---

### 4. Elliptic Curve Digital Signature Scheme (ECDSS) Implemented CPU only:

Implements both **signature generation** and **verification** in hardware and Python.

# A. Generate ECC keypair (Private and Public keys)
- d, e1, e2 = generate_keys()
- msg = "ECC IS AWESOME"

# B. Sign message using private key
- S1, S2 = sign_message(msg, d)
- print("\n=== Signature ===")
- print(f"S1 = {hex(S1)}")
- print(f"S2 = {hex(S2)}")

# C. Verify signature using public key
- valid = verify_signature(msg, S1, S2, e2)
- print("\n=== Verification ===")
- print("Valid signature ✅" if valid else "Invalid ❌")

---

## 🧮 Curve Parameters

All curve parameters conform to **NIST FIPS 186-4**.

### GF(2¹⁶³) — sect163r2
a = 1
b = 0x20A601907B8C953CA1481EB10512F78744A3205FD

### GF(2²³³) — sect233r1
a = 1
b = 0x0x066647ede6c332c7f8c0923bb58213b333b20e9ce4281fe115f7d8f90ad

### GF(2⁵⁷¹) — sect571r1
a = 1
b = 0x02f40e7e2221f295de297117b7f3d62f5c6a97ffcb8ceff1cd6ba8ce4a9a18ad84ffabbd8efa59332be7ad6756a66e294afd185a78ff12aa520e4de739baca0c7ffeff7f2955727a

---

## 🧩 Features

✅ Supports multiple ECC binary fields (GF(2¹⁶³), GF(2²³³), GF(2⁵⁷¹))  
✅ Verilog modules for FPGA or ASIC  
✅ Python-generated Verilog testbenches  
✅ Consistent scalar multiplication and ECDSS implementations  
✅ Hardware-efficient field arithmetic (Karatsuba–Ofman, Itoh–Tsujii)  

---

## 🧰 Tools and Dependencies

### Hardware
- **Language:** Verilog HDL  
- **Simulators:** Vivado iSimulator 
- **FPGA Tools:** Xilinx Vivado 

### Software
- **Python ≥ 3.8**
  - numpy
  - Cupy lib (Cuda kernals)
  - sympy
  - bitstring

---

## 🧠 Mathematical Foundations

1. **Lopez–Dahab Projective Coordinates**  
2. **Itoh–Tsujii Inversion Algorithm**  
3. **Karatsuba–Ofman Multiplication**  
