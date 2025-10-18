`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 117-bit × 117-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult58 + mult59
// Passes clock into lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult117 (
    input  wire        clk,
    input  wire [116:0] a,
    input  wire [116:0] b,
    output wire [233:0] d
);

    // Split: 58 (low) + 59 (high)
    wire [57:0] a_low  = a[57:0];
    wire [58:0] a_high = a[116:58];
    wire [57:0] b_low  = b[57:0];
    wire [58:0] b_high = b[116:58];

    wire [115:0] z0;   // mult58
    wire [117:0] z2;   // mult59
    wire [117:0] z1;   // mult59

    // z0 = mult58(a_low, b_low)
    mult58 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult59(a_high, b_high)
    mult59 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult59((a_low ^ a_high), (b_low ^ b_high))
    mult59 u_z1 (
        .clk(clk),
        .a({1'b0, a_low} ^ a_high),  // promote to 59 bits
        .b({1'b0, b_low} ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 116) ^ ((z1 ^ z2 ^ z0) << 58) ^ z0
    assign d = ({z2, 116'b0}) ^
               ({(z1 ^ z2 ^ {2'b00, z0}), 58'b0}) ^
               {118'b0, z0};

endmodule
