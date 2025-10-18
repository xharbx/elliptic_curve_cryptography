`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2^233) Reduction
// Modulo polynomial: P(x) = x^233 + x^74 + 1
// Input:  465-bit raw product (from mult233)
// Output: 233-bit reduced result
//////////////////////////////////////////////////////////////////////////////////
module gf2_reduce_233 (
    input  wire [464:0] in,
    output wire [232:0] out
);
    reg [464:0] poly;
    integer i;

    always @(*) begin
        poly = in;
        // While degree >= 233, fold down
        for (i = 464; i >= 233; i = i - 1) begin
            if (poly[i]) begin
                poly[i] = 1'b0;                // clear bit
                poly[i-233] = poly[i-233] ^ 1'b1;   // fold into x^k
                poly[i-159] = poly[i-159] ^ 1'b1;   // fold into x^(k+74)
            end
        end
    end

    assign out = poly[232:0];
endmodule

