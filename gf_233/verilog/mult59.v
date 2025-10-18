`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 59-bit × 59-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult29 + mult30
// Passes clock into lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult59 (
    input  wire        clk,
    input  wire [58:0] a,
    input  wire [58:0] b,
    output wire [117:0] d
);

    // Split into 29 (low) + 30 (high)
    wire [28:0] a_low  = a[28:0];
    wire [29:0] a_high = a[58:29];
    wire [28:0] b_low  = b[28:0];
    wire [29:0] b_high = b[58:29];

    wire [57:0] z0;   // mult29
    wire [59:0] z2;   // mult30
    wire [59:0] z1;   // cross term

    // z0 = mult29(a_low, b_low)
    mult29 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult30(a_high, b_high)
    mult30 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult30((a_low ^ a_high), (b_low ^ b_high)), promoted to 30 bits
    mult30 u_z1 (
        .clk(clk),
        .a({1'b0, a_low} ^ a_high),
        .b({1'b0, b_low} ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 58) ^ ((z1 ^ z2 ^ z0) << 29) ^ z0
    assign d = ({z2, 58'b0}) ^
               ({(z1 ^ z2 ^ {2'b00, z0}), 29'b0}) ^
               {60'b0, z0};

endmodule
