`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 571-bit × 571-bit
// Polynomial multiplication (no reduction)
// Implemented with Karatsuba recursion using mult285 + mult286
// Passes clock into lower modules
//////////////////////////////////////////////////////////////////////////////////
module mult571 (
    input  wire        clk,
    input  wire [570:0] a,
    input  wire [570:0] b,
    output wire [1141:0] d
);

    // Split into 285 (low) + 286 (high)
    wire [284:0] a_low  = a[284:0];
    wire [285:0] a_high = a[570:285];
    wire [284:0] b_low  = b[284:0];
    wire [285:0] b_high = b[570:285];

    wire [569:0] z0;   // mult285
    wire [571:0] z2;   // mult286
    wire [571:0] z1;   // mult286

    // z0 = mult285(a_low, b_low)
    mult285 u_z0 (
        .clk(clk),
        .a(a_low),
        .b(b_low),
        .d(z0)
    );

    // z2 = mult286(a_high, b_high)
    mult286 u_z2 (
        .clk(clk),
        .a(a_high),
        .b(b_high),
        .d(z2)
    );

    // z1 = mult286(a_low ^ a_high, b_low ^ b_high)
    mult286 u_z1 (
        .clk(clk),
        .a({1'b0, a_low} ^ a_high),  // promote to 286 bits
        .b({1'b0, b_low} ^ b_high),
        .d(z1)
    );

    // Karatsuba recombination:
    // d = (z2 << 570) ^ ((z1 ^ z2 ^ z0) << 285) ^ z0
    assign d = ({z2, 570'b0}) ^
               ({(z1 ^ z2 ^ {2'b0, z0}), 285'b0}) ^
               {570'b0, z0};

endmodule
