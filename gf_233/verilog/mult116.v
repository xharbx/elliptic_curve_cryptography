`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 116-bit × 116-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult58
// Passes clock into lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult116 (
    input  wire        clk,
    input  wire [115:0] a,
    input  wire [115:0] b,
    output wire [231:0] d
);

    // Split into two 58-bit halves
    wire [57:0] a_low  = a[57:0];
    wire [57:0] a_high = a[115:58];
    wire [57:0] b_low  = b[57:0];
    wire [57:0] b_high = b[115:58];

    wire [115:0] z0;   // mult58
    wire [115:0] z2;   // mult58
    wire [115:0] z1;   // mult58

    // z0 = mult58(a_low, b_low)
    mult58 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult58(a_high, b_high)
    mult58 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult58(a_low ^ a_high, b_low ^ b_high)
    mult58 u_z1 (
        .clk(clk),
        .a(a_low ^ a_high),
        .b(b_low ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 116) ^ ((z1 ^ z2 ^ z0) << 58) ^ z0
    assign d = ({z2, 116'b0}) ^
               ({(z1 ^ z2 ^ z0), 58'b0}) ^
               {116'b0, z0};

endmodule
