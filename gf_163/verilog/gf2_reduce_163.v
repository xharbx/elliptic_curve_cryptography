`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2^163) Reduction
// Polynomial: P(x) = x^163 + x^7 + x^6 + x^3 + 1
// Input:  up to 326-bit raw product
// Output: 163-bit reduced result
//////////////////////////////////////////////////////////////////////////////////
module gf2_reduce_163 (
    input  wire [325:0] in,
    output wire [162:0] out
);
    reg [325:0] tmp;
    integer k;

    always @* begin
        tmp = in;
        // Fold down terms above x^162
        for (k = 325; k >= 163; k = k - 1) begin
            if (tmp[k]) begin
                tmp[k]      = 1'b0;
                // Replace x^k with x^(k-163) + x^(k-163+7) + x^(k-163+6) + x^(k-163+3)
                tmp[k-163]  = tmp[k-163] ^ 1'b1;
                tmp[k-156]  = tmp[k-156] ^ 1'b1; // +7
                tmp[k-157]  = tmp[k-157] ^ 1'b1; // +6
                tmp[k-160]  = tmp[k-160] ^ 1'b1; // +3
            end
        end
    end

    assign out = tmp[162:0];

endmodule
