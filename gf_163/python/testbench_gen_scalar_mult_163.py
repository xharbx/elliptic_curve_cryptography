# -*- coding: utf-8 -*-
"""
Auto-generate Verilog Testbench for EC Scalar Multiplication over GF(2^163)
Curve: sect163r2 (Lopez–Dahab projective coordinates)
Golden reference: ecc_gf2m_163(k, G) in Python
"""

import random

# === Import your golden reference model ===
from scalar_mult_163 import ecc_gf2m_163, Gx, Gy   

# === Parameters ===
NUM_TESTS = 500

def generate_test_vectors(num_tests=NUM_TESTS):
    """Generate random scalars plus edge cases"""
    tests = [1, 2, 3, 5, 10, 120, 255, (1 << 162), (1 << 163) - 1]
    for _ in range(num_tests - len(tests)):
        tests.append(random.getrandbits(163))
    return tests


def generate_tb(filename=f"tb_ec_scalar_mult_163_num_points_{NUM_TESTS}.v", num_tests=NUM_TESTS):
    vectors = generate_test_vectors(num_tests)
    results = []

    # Base point G (projective input)
    G = (Gx, Gy, 1)

    print("Generating", num_tests, "test vectors...")
    for k in vectors:
        x, y = ecc_gf2m_163(k, G)  # golden affine result
        results.append((k, x, y))

    with open(filename, "w") as f:
        f.write("`timescale 1ns / 1ps\n")
        f.write("//////////////////////////////////////////////////////////////////////////////////\n")
        f.write("// Auto-generated Testbench for EC Scalar Multiplication GF(2^163)\n")
        f.write("// Curve: sect163r2\n")
        f.write("//////////////////////////////////////////////////////////////////////////////////\n\n")
        f.write("module tb_ec_scalar_mult_163;\n\n")
        f.write("  reg clk, rst, start;\n")
        f.write("  reg  [162:0] k;\n")
        f.write("  reg  [162:0] Px, Py;\n")
        f.write("  wire [162:0] X, Y;\n")
        f.write("  wire done;\n\n")

        f.write("  // Instantiate DUT\n")
        f.write("  ec_scalar_mult_163 DUT (\n")
        f.write("    .clk(clk), .rst(rst), .start(start),\n")
        f.write("    .k(k), .Px(Px), .Py(Py),\n")
        f.write("    .X(X), .Y(Y), .done(done)\n")
        f.write("  );\n\n")

        f.write("  // Clock\n")
        f.write("  always #5 clk = ~clk;\n\n")

        f.write(f"  reg [162:0] test_k [0:{num_tests-1}];\n")
        f.write(f"  reg [162:0] exp_x  [0:{num_tests-1}];\n")
        f.write(f"  reg [162:0] exp_y  [0:{num_tests-1}];\n")
        f.write("  integer i;\n\n")

        f.write("  initial begin\n")
        f.write("    clk   = 0;\n")
        f.write("    rst   = 1;\n")
        f.write("    start = 0;\n")
        f.write("    #20; rst = 0;\n\n")

        # Preload test vectors
        for idx, (kval, xval, yval) in enumerate(results):
            f.write(f"    test_k[{idx}] = 163'h{kval:x};\n")
            f.write(f"    exp_x[{idx}]  = 163'h{xval:x};\n")
            f.write(f"    exp_y[{idx}]  = 163'h{yval:x};\n")

        f.write("\n    for (i=0; i<" + str(num_tests) + "; i=i+1) begin\n")
        f.write("      @(negedge clk);\n")
        f.write("      k = test_k[i];\n")
        f.write("      Px = 163'h3f0eba16286a2d57ea0991168d4994637e8343e36;\n")
        f.write("      Py = 163'h0d51fbc6c71a0094fa2cdd545b11c5c0c797324f1;\n")
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
