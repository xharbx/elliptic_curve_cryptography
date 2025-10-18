`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2^571) Multiplier (with reduction)
// Polynomial: P(x) = x^571 + x^10 + x^5 + x^2 + 1
//////////////////////////////////////////////////////////////////////////////////
module gf2m_mult571 (
    input  wire         clk,   // optional pipeline clock
    input  wire [570:0] a,
    input  wire [570:0] b,
    output wire  [570:0] c
);

    // Step 1: Raw Karatsuba multiplication (571 x 571 -> 1142 bits)
    wire [1141:0] raw_comb;
    reg  [1141:0] raw_reg;
    
    mult571 u_mult571 (
        .clk(clk),   // <<< missing connection
        .a(a),
        .b(b),
        .d(raw_comb)
    );
    
    

    // Register stage for raw product
    always @(posedge clk) begin
        raw_reg <= raw_comb;
    end

    // Step 2: Modular reduction (combinational)
    gf2_reduce_571 u_reduce (
        .in(raw_reg),
        .out(c)
    );


endmodule
