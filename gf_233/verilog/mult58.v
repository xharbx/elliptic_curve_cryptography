`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 58-bit × 58-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult29
// Passes clock to lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult58 (
    input  wire        clk,
    input  wire [57:0] a,
    input  wire [57:0] b,
    output wire [115:0] d
);

    // Split into two 29-bit halves
    wire [28:0] a_low  = a[28:0];
    wire [28:0] a_high = a[57:29];
    wire [28:0] b_low  = b[28:0];
    wire [28:0] b_high = b[57:29];

    wire [57:0] z0;   // mult29
    wire [57:0] z2;   // mult29
    wire [57:0] z1;   // mult29

    // z0 = mult29(a_low, b_low)
    mult29 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult29(a_high, b_high)
    mult29 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult29(a_low ^ a_high, b_low ^ b_high)
    mult29 u_z1 (
        .clk(clk),
        .a(a_low ^ a_high),
        .b(b_low ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 58) ^ ((z1 ^ z2 ^ z0) << 29) ^ z0
    assign d = ({z2, 58'b0}) ^ ({(z1 ^ z2 ^ z0), 29'b0}) ^ {58'b0, z0};

endmodule
