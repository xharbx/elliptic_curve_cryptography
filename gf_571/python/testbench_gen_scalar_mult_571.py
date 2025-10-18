# -*- coding: utf-8 -*-
"""
Auto-generate Verilog Testbench for EC Scalar Multiplication over GF(2^571)
Curve: sect571r1 (Lopez–Dahab projective coordinates)
Golden reference: ecc_gf2m_571(k, G) in Python
"""

import random

# === Import your golden reference model ===
from scalar_mult_571 import ecc_gf2m_571, Gx, Gy   # <-- put your scalar_mult code in ecc571_model.py

# === Parameters ===
NUM_TESTS = 20

def generate_test_vectors(num_tests=NUM_TESTS):
    """Generate random scalars plus edge cases"""
    tests = [1, 2, 3, 5, 10, 120, 255, (1 << 570), (1 << 571) - 1]
    for _ in range(num_tests - len(tests)):
        tests.append(random.getrandbits(571))
    return tests


def generate_tb(filename= f"tb_ec_scalar_mult_571_num_points_{NUM_TESTS}.v", num_tests=NUM_TESTS):
    vectors = generate_test_vectors(num_tests)
    results = []

    # Base point G (projective input)
    G = (Gx, Gy, 1)

    print("Generating", num_tests, "test vectors...")
    for k in vectors:
        x, y = ecc_gf2m_571(k, G)  # golden affine result
        results.append((k, x, y))

    with open(filename, "w") as f:
        f.write("`timescale 1ns / 1ps\n")
        f.write("//////////////////////////////////////////////////////////////////////////////////\n")
        f.write("// Auto-generated Testbench for EC Scalar Multiplication GF(2^571)\n")
        f.write("// Curve: sect571r1\n")
        f.write("//////////////////////////////////////////////////////////////////////////////////\n\n")
        f.write("module tb_ec_scalar_mult_571;\n\n")
        f.write("  reg clk, rst, start;\n")
        f.write("  reg  [570:0] k;\n")
        f.write("  reg  [570:0] Px, Py;\n")
        f.write("  wire [570:0] X, Y;\n")
        f.write("  wire done;\n\n")

        f.write("  // Instantiate DUT\n")
        f.write("  ec_scalar_mult_571 DUT (\n")
        f.write("    .clk(clk), .rst(rst), .start(start),\n")
        f.write("    .k(k), .Px(Px), .Py(Py),\n")
        f.write("    .X(X), .Y(Y), .done(done)\n")
        f.write("  );\n\n")

        f.write("  // Clock\n")
        f.write("  always #5 clk = ~clk;\n\n")

        f.write(f"  reg [570:0] test_k [0:{num_tests-1}];\n")
        f.write(f"  reg [570:0] exp_x  [0:{num_tests-1}];\n")
        f.write(f"  reg [570:0] exp_y  [0:{num_tests-1}];\n")
        f.write("  integer i;\n\n")

        f.write("  initial begin\n")
        f.write("    clk   = 0;\n")
        f.write("    rst   = 1;\n")
        f.write("    start = 0;\n")
        f.write("    #20; rst = 0;\n\n")

        # Preload test vectors
        for idx, (kval, xval, yval) in enumerate(results):
            f.write(f"    test_k[{idx}] = 571'h{kval:x};\n")
            f.write(f"    exp_x[{idx}]  = 571'h{xval:x};\n")
            f.write(f"    exp_y[{idx}]  = 571'h{yval:x};\n")

        f.write("\n    for (i=0; i<" + str(num_tests) + "; i=i+1) begin\n")
        f.write("      @(negedge clk);\n")
        f.write("      k = test_k[i];\n")
        
        f.write("      Px = 571'h303001d34b856296c16c0d40d3cd7750a93d1d2955fa80aa5f40fc8db7b2abdbde53950f4c0d293cdd711a35b67fb1499ae60038614f1394abfa3b4c850d927e1e7769c8eec2d19;\n")
        f.write("      Py = 571'h37bf27342da639b6dccfffeb73d69d78c6c27a6009cbbca1980f8533921e8a684423e43bab08a576291af8f461bb2a8b3531d2f0485c19b16e2f1516e23dd3c1a4827af1b8ac15b;\n")
        
        f.write("      start = 1;\n")
        f.write("      @(negedge clk); start = 0;\n")
        f.write("      wait(done);\n\n")

        f.write("      if (X !== exp_x[i] || Y !== exp_y[i]) begin\n")
        f.write("        $display(\"FAIL: k=%h, got X=%h Y=%h, expected X=%h Y=%h\", k, X, Y, exp_x[i], exp_y[i]);\n")
        f.write("        $stop;\n")
        f.write("      end else begin\n")
        f.write("        $display(\"PASS: k=%h\", k);\n")
        f.write("      end\n")
        f.write("      #20;\n")
        f.write("    end\n")
        f.write("    $display(\"All scalar multiplication tests PASSED.\");\n")
        f.write("    $finish;\n")
        f.write("  end\n")
        f.write("endmodule\n")

    print(f"Generated {filename} with {num_tests} test vectors.")


if __name__ == "__main__":
    generate_tb()
