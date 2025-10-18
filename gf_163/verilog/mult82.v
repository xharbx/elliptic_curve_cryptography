`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 82-bit × 82-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult41
// Clock is passed down to lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult82 (
    input  wire        clk,
    input  wire [81:0] a,
    input  wire [81:0] b,
    output wire [163:0] d
);

    // Split into two 41-bit halves
    wire [40:0] a_low  = a[40:0];
    wire [40:0] a_high = a[81:41];
    wire [40:0] b_low  = b[40:0];
    wire [40:0] b_high = b[81:41];

    wire [81:0] z0;   // mult41
    wire [81:0] z2;   // mult41
    wire [81:0] z1;   // mult41

    // z0 = mult41(a_low, b_low)
    mult41 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult41(a_high, b_high)
    mult41 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult41(a_low ^ a_high, b_low ^ b_high)
    mult41 u_z1 (
        .clk(clk),
        .a(a_low ^ a_high),
        .b(b_low ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 82) ^ ((z1 ^ z2 ^ z0) << 41) ^ z0
    assign d = ({z2, 82'b0}) ^
               ({(z1 ^ z2 ^ z0), 41'b0}) ^
               {82'b0, z0};

endmodule
