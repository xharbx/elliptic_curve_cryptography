`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 143-bit × 143-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult71 + mult72
// Passes clock into lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult143 (
    input  wire        clk,
    input  wire [142:0] a,
    input  wire [142:0] b,
    output wire [285:0] d
);

    // Split into 71 (low) + 72 (high)
    wire [70:0] a_low  = a[70:0];
    wire [71:0] a_high = a[142:71];
    wire [70:0] b_low  = b[70:0];
    wire [71:0] b_high = b[142:71];

    wire [141:0] z0;   // mult71
    wire [143:0] z2;   // mult72
    wire [143:0] z1;   // cross term

    // z0 = mult71(a_low, b_low)
    mult71 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult72(a_high, b_high)
    mult72 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult72((a_low ^ a_high), (b_low ^ b_high)), promoted to 72 bits
    mult72 u_z1 (
        .clk(clk),
        .a({1'b0, a_low} ^ a_high),
        .b({1'b0, b_low} ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 142) ^ ((z1 ^ z2 ^ z0) << 71) ^ z0
    assign d = ({z2, 142'b0}) ^
               ({(z1 ^ z2 ^ {2'b0, z0}), 71'b0}) ^
               {144'b0, z0};

endmodule
