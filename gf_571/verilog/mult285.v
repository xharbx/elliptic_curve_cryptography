`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 285-bit × 285-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult142 + mult143
// Passes clock into lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult285 (
    input  wire        clk,
    input  wire [284:0] a,
    input  wire [284:0] b,
    output wire [569:0] d
);

    // Split into 142 (low) + 143 (high)
    wire [141:0] a_low  = a[141:0];
    wire [142:0] a_high = a[284:142];
    wire [141:0] b_low  = b[141:0];
    wire [142:0] b_high = b[284:142];

    wire [283:0] z0;   // mult142
    wire [285:0] z2;   // mult143
    wire [285:0] z1;   // mult143

    // z0 = mult142(a_low, b_low)
    mult142 u_z0 (
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
        .a({1'b0, a_low} ^ a_high),  // promote to 143 bits
        .b({1'b0, b_low} ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 284) ^ ((z1 ^ z2 ^ z0) << 142) ^ z0
    assign d = ({z2, 284'b0}) ^
               ({(z1 ^ z2 ^ {2'b0, z0}), 142'b0}) ^
               {284'b0, z0};

endmodule
