`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2^571) Reduction
// P(x) = x^571 + x^10 + x^5 + x^2 + 1
// Input:  up to 1142-bit raw product
// Output: 571-bit reduced result
//////////////////////////////////////////////////////////////////////////////////
module gf2_reduce_571 (
    input  wire [1141:0] in,
    output wire [570:0]  out
);
    reg [1141:0] tmp;
    integer k;

    always @* begin
        tmp = in;
        // Fold down terms above x^570
        for (k = 1141; k >= 571; k = k - 1) begin
            if (tmp[k]) begin
                tmp[k]      = 1'b0;
                // Replace x^k with x^(k-571) + x^(k-571+10) + x^(k-571+5) + x^(k-571+2)
                tmp[k-571]  = tmp[k-571] ^ 1'b1;
                tmp[k-561]  = tmp[k-561] ^ 1'b1; // (k-571)+10
                tmp[k-566]  = tmp[k-566] ^ 1'b1; // (k-571)+5
                tmp[k-569]  = tmp[k-569] ^ 1'b1; // (k-571)+2
            end
        end
    end

    assign out = tmp[570:0];

endmodule
