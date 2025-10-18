`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2^163) Multiplier (with reduction)
// Polynomial: P(x) = x^163 + x^7 + x^6 + x^3 + 1
//////////////////////////////////////////////////////////////////////////////////
module gf2m_mult163 (
    input  wire         clk,
    input  wire [162:0] a,
    input  wire [162:0] b,
    output wire [162:0] c
);

    // Step 1: Raw Karatsuba multiplication (163 x 163 -> 326 bits)
    wire [325:0] raw_comb;
    reg  [325:0] raw_reg;

    mult163 u_mult163 (
        .clk(clk),
        .a(a),
        .b(b),
        .d(raw_comb)
    );

    // Register stage for raw product
    always @(posedge clk) begin
        raw_reg <= raw_comb;
    end

    // Step 2: Modular reduction (combinational)
    gf2_reduce_163 u_reduce (
        .in(raw_reg),
        .out(c)
    );

endmodule
