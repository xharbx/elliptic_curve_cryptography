`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 41-bit × 41-bit
// Polynomial multiplication, no reduction
// Implemented with Karatsuba recursion using mult20 + mult21
// Clock is passed down to leaf multipliers
//////////////////////////////////////////////////////////////////////////////////
module mult41 (
    input  wire        clk,
    input  wire [40:0] a,
    input  wire [40:0] b,
    output wire [81:0] d
);

    // Split into 20 (low) + 21 (high)
    wire [19:0] a_low  = a[19:0];
    wire [20:0] a_high = a[40:20];
    wire [19:0] b_low  = b[19:0];
    wire [20:0] b_high = b[40:20];

    wire [39:0] z0;   // mult20
    wire [41:0] z2;   // mult21
    wire [41:0] z1;   // cross term

    // z0 = mult20(a_low, b_low)
    mult20 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult21(a_high, b_high)
    mult21 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult21((a_low ^ a_high), (b_low ^ b_high)), promote to 21 bits
    mult21 u_z1 (
        .clk(clk),
        .a({1'b0, a_low} ^ a_high),
        .b({1'b0, b_low} ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 40) ^ ((z1 ^ z2 ^ z0) << 20) ^ z0
    assign d = ({z2, 40'b0}) ^
               ({(z1 ^ z2 ^ {2'b0, z0}), 20'b0}) ^
               {42'b0, z0};

endmodule
