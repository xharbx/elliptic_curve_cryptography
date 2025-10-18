`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 40-bit × 40-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult20
// Clock is passed down to leaves
//////////////////////////////////////////////////////////////////////////////////
module mult40 (
    input  wire        clk,
    input  wire [39:0] a,
    input  wire [39:0] b,
    output wire [79:0] d
);

    // Split into two 20-bit halves
    wire [19:0] a_low  = a[19:0];
    wire [19:0] a_high = a[39:20];
    wire [19:0] b_low  = b[19:0];
    wire [19:0] b_high = b[39:20];

    wire [39:0] z0;   // mult20
    wire [39:0] z2;   // mult20
    wire [39:0] z1;   // mult20

    // z0 = mult20(a_low, b_low)
    mult20 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult20(a_high, b_high)
    mult20 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult20(a_low ^ a_high, b_low ^ b_high)
    mult20 u_z1 (
        .clk(clk),
        .a(a_low ^ a_high),
        .b(b_low ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 40) ^ ((z1 ^ z2 ^ z0) << 20) ^ z0
    assign d = ({z2, 40'b0}) ^
               ({(z1 ^ z2 ^ z0), 20'b0}) ^
               {40'b0, z0};

endmodule
