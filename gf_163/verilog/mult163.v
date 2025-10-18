`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 163-bit × 163-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult81 + mult82
// Clock is passed into lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult163 (
    input  wire        clk,
    input  wire [162:0] a,
    input  wire [162:0] b,
    output wire [325:0] d
);

    // Split into 81 (low) + 82 (high)
    wire [80:0] a_low  = a[80:0];
    wire [81:0] a_high = a[162:81];
    wire [80:0] b_low  = b[80:0];
    wire [81:0] b_high = b[162:81];

    wire [161:0] z0;   // mult81
    wire [163:0] z2;   // mult82
    wire [163:0] z1;   // mult82

    // z0 = mult81(a_low, b_low)
    mult81 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult82(a_high, b_high)
    mult82 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult82(a_low ^ a_high, b_low ^ b_high), promote low halves
    mult82 u_z1 (
        .clk(clk),
        .a({1'b0, a_low} ^ a_high),
        .b({1'b0, b_low} ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 162) ^ ((z1 ^ z2 ^ z0) << 81) ^ z0
    assign d = ({z2, 162'b0}) ^
               ({(z1 ^ z2 ^ {2'b0, z0}), 81'b0}) ^
               {162'b0, z0};

endmodule
