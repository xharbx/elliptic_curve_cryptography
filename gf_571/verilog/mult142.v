`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 142-bit × 142-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult71
// Passes clock into lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult142 (
    input  wire        clk,
    input  wire [141:0] a,
    input  wire [141:0] b,
    output wire [283:0] d
);

    // Split into two 71-bit halves
    wire [70:0] a_low  = a[70:0];
    wire [70:0] a_high = a[141:71];
    wire [70:0] b_low  = b[70:0];
    wire [70:0] b_high = b[141:71];

    wire [141:0] z0;   // mult71
    wire [141:0] z2;   // mult71
    wire [141:0] z1;   // mult71

    // z0 = mult71(a_low, b_low)
    mult71 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult71(a_high, b_high)
    mult71 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult71(a_low ^ a_high, b_low ^ b_high)
    mult71 u_z1 (
        .clk(clk),
        .a(a_low ^ a_high),
        .b(b_low ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 142) ^ ((z1 ^ z2 ^ z0) << 71) ^ z0
    assign d = ({z2, 142'b0}) ^
               ({(z1 ^ z2 ^ z0), 71'b0}) ^
               {142'b0, z0};

endmodule
