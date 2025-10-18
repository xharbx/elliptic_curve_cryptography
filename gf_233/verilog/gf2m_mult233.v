`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2^233) Multiplier (with reduction)
// P(x) = x^233 + x^74 + 1
// Registers the output of mult233 before reduction
//////////////////////////////////////////////////////////////////////////////////
module gf2m_mult233 (
    input  wire        clk,
    input  wire [232:0] a,
    input  wire [232:0] b,
    output wire  [232:0] c
);

    wire [464:0] raw_comb;
    reg  [464:0] raw_reg;

    // Step 1: Karatsuba multiplication (pipelined)
    mult233 u_mult233 (
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
    gf2_reduce_233 u_reduce (
        .in(raw_reg),
        .out(c)
    );


endmodule
