`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 286-bit × 286-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult143
// Passes clock into lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult286 (
    input  wire        clk,
    input  wire [285:0] a,
    input  wire [285:0] b,
    output wire [571:0] d
);

    // Split into two 143-bit halves
    wire [142:0] a_low  = a[142:0];
    wire [142:0] a_high = a[285:143];
    wire [142:0] b_low  = b[142:0];
    wire [142:0] b_high = b[285:143];

    wire [285:0] z0;   // mult143
    wire [285:0] z2;   // mult143
    wire [285:0] z1;   // mult143

    // z0 = mult143(a_low, b_low)
    mult143 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult143(a_high, b_high)
    mult143 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult143(a_low ^ a_high, b_low ^ b_high)
    mult143 u_z1 (
        .clk(clk),
        .a(a_low ^ a_high),
        .b(b_low ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 286) ^ ((z1 ^ z2 ^ z0) << 143) ^ z0
    assign d = ({z2, 286'b0}) ^
               ({(z1 ^ z2 ^ z0), 143'b0}) ^
               {286'b0, z0};

endmodule
