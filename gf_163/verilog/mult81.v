`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 81-bit × 81-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult40 + mult41
// Clock is passed down to lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult81 (
    input  wire        clk,
    input  wire [80:0] a,
    input  wire [80:0] b,
    output wire [161:0] d
);

    // Split into 40 (low) + 41 (high)
    wire [39:0] a_low  = a[39:0];
    wire [40:0] a_high = a[80:40];
    wire [39:0] b_low  = b[39:0];
    wire [40:0] b_high = b[80:40];

    wire [79:0]  z0;   // mult40
    wire [81:0]  z2;   // mult41
    wire [81:0]  z1;   // mult41

    // z0 = mult40(a_low, b_low)
    mult40 u_z0 (
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

    // z1 = mult41((a_low ^ a_high), (b_low ^ b_high)), promote low to 41 bits
    mult41 u_z1 (
        .clk(clk),
        .a({1'b0, a_low} ^ a_high),
        .b({1'b0, b_low} ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 80) ^ ((z1 ^ z2 ^ z0) << 40) ^ z0
    assign d = ({z2, 80'b0}) ^
               ({(z1 ^ z2 ^ {2'b0, z0}), 40'b0}) ^
               {82'b0, z0};

endmodule
