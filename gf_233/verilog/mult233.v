`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 233-bit × 233-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult116 + mult117
// Passes clock into lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult233 (
    input  wire        clk,
    input  wire [232:0] a,
    input  wire [232:0] b,
    output wire [464:0] d
);

    // Split into 116 (low) + 117 (high)
    wire [115:0] a_low  = a[115:0];
    wire [116:0] a_high = a[232:116];
    wire [115:0] b_low  = b[115:0];
    wire [116:0] b_high = b[232:116];

    wire [231:0] z0;   // mult116
    wire [233:0] z2;   // mult117
    wire [233:0] z1;   // mult117

    // z0 = mult116(a_low, b_low)
    mult116 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult117(a_high, b_high)
    mult117 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult117((a_low ^ a_high), (b_low ^ b_high))
    mult117 u_z1 (
        .clk(clk),
        .a({1'b0, a_low} ^ a_high),  // promote to 117 bits
        .b({1'b0, b_low} ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 232) ^ ((z1 ^ z2 ^ z0) << 116) ^ z0
    assign d = ({z2, 232'b0}) ^
               ({(z1 ^ z2 ^ {2'b00, z0}), 116'b0}) ^
               {232'b0, z0};

endmodule
